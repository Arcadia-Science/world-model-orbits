"""Model training and rollout-evaluation entrypoint.

Builds bias-free (ELM-compatible) GPT models, trains them on the Kepler
dataset with optional noisy-context augmentation, periodically probes the
learned representations, and saves checkpoints. Run as a CLI:

    python -m analysis.training
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score

from .model import GPTConfigCV, GPTCV
from .data import load_trajectories, chop_trajectories_into_sequences, load_orbital_params
from .probing import (
    setup_activation_hooks, collect_activations, run_linear_probes, run_geometry_probes,
)

# Reproducibility + device (training entrypoint only)
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

num_points_per_trajectory = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gpu_memory_stats():
    """Get GPU memory usage statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated,
        }
    return {'allocated_gb': 0, 'reserved_gb': 0, 'max_allocated_gb': 0}


def print_gpu_memory_stats(label=""):
    """Print GPU memory usage statistics."""
    stats = get_gpu_memory_stats()
    if torch.cuda.is_available():
        print(f"{label}GPU Memory - Allocated: {stats['allocated_gb']:.3f} GB, "
              f"Reserved: {stats['reserved_gb']:.3f} GB, "
              f"Max Allocated: {stats['max_allocated_gb']:.3f} GB")
    return stats


def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def setup_model(block_size, n_layer=2, n_embd=32, device=None, variable_dt=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if variable_dt:
        # Variable-dt model: input token (x, y, dt) -> predict next (x, y). Uses
        # the sister module model_cv_detached_dt.py (decoupled input/output dims).
        # Standard runs stay on model_cv_detached.py so their checkpoints and the
        # existing analysis tooling are untouched.
        # NOTE: variable-dt models are not part of this publication's ported scope;
        # model_cv_detached_dt is intentionally absent, so this branch is unsupported.
        from .model_cv_detached_dt import GPTConfigCV as GPTConfigCVDT, GPTCV as GPTCVDT
        config = GPTConfigCVDT(
            block_size=block_size,
            input_dim=3,
            output_dim=2,
            n_layer=n_layer,
            n_head=1,
            n_embd=n_embd,
            attention_alpha=0.0,
            bias=False,
            dropout=0.0,
        )
        model = GPTCVDT(config)
    else:
        config = GPTConfigCV(
            block_size=block_size,
            input_dim=2,
            n_layer=n_layer,
            n_head=1,
            n_embd=n_embd,
            attention_alpha=0.0,
            bias=False,
            dropout=0.0,
        )
        model = GPTCV(config)
    model = model.to(device)
    return model


def compute_loss_with_mask(predictions, targets, loss_mask='all'):
    """
    Compute MSE loss with optional masking.

    Args:
        predictions: tensor of shape (batch, seq_len, 2)
        targets: tensor of shape (batch, target_len, 2)
        loss_mask: 'all' to compute loss on all positions, 'last' to compute only on last position

    Returns:
        loss: scalar tensor
    """
    # Handle target shape: if 2D, add dimension
    if targets.dim() == 2:
        targets = targets.unsqueeze(-1)  # (b, t) -> (b, t, 1)

    # Handle case where targets might have different sequence length than predictions
    if targets.size(1) < predictions.size(1):
        # Only use the last predictions that match target length
        predictions = predictions[:, -targets.size(1):, :]  # Take last t_target positions
    elif targets.size(1) > predictions.size(1):
        raise ValueError(f"Target sequence length {targets.size(1)} > prediction length {predictions.size(1)}")

    if loss_mask == 'all':
        # Compute loss on all positions
        predictions_flat = predictions.reshape(-1)  # shape (b*t*d,)
        targets_flat = targets.reshape(-1)  # shape (b*t*d,)
        loss = F.mse_loss(predictions_flat, targets_flat)
    elif loss_mask == 'last':
        # Compute loss only on the last position
        predictions_last = predictions[:, -1, :].reshape(-1)  # shape (b*d,)
        targets_last = targets[:, -1, :].reshape(-1)  # shape (b*d,)
        loss = F.mse_loss(predictions_last, targets_last)
    else:
        raise ValueError(f"loss_mask must be 'all' or 'last', got {loss_mask}")

    return loss


def rollout_predict(model, x0, horizon, block_size):
    """
    Differentiable autoregressive rollout for the rollout-loss objective.

    Starting from the (already noised, if any) input window x0 of shape
    (batch, block_size, 2), generate `horizon` future points one at a time,
    feeding each prediction back in (sliding the last `block_size` points). The
    model is NOT given the architecture of the dynamics -- it must produce a
    self-consistent multi-step rollout on its own; gradients flow through the
    whole chain so the model is penalised for compounding (e.g. energy-drift)
    error, not just one-step error.

    Returns predictions of shape (batch, horizon, 2).
    """
    seq = x0
    preds = []
    for _ in range(horizon):
        out, _ = model.forward(seq[:, -block_size:, :], None)
        nxt = out[:, -1:, :]                 # last-position next-step prediction
        preds.append(nxt)
        seq = torch.cat([seq, nxt], dim=1)
    return torch.cat(preds, dim=1)


def train_model(model, train_inputs, train_targets, test_inputs, test_targets, train_trajectories, test_trajectories,
                 n_steps=1001, lr=1e-3, weight_decay=0.0, noise_scale=0.1, prob_freq=100, batch_size=128,
                 loss_mask='all', seed=1, orbital_params=None, rollout_horizon=1, variable_dt=False):
    """
    Train the model on the trajectory data with periodic evaluation.

    Args:
        model: Model to train
        train_inputs: Training input trajectories
        train_targets: Training target trajectories
        test_inputs: Test input trajectories
        test_targets: Test target trajectories
        train_trajectories: Full original training trajectories for evaluation
        test_trajectories: Full original test trajectories for evaluation
        n_steps: Number of training steps
        lr: Learning rate
        weight_decay: Weight decay
        noise_scale: Scale of noise added during training
        prob_freq: Frequency of evaluation (every N steps)
        batch_size: Batch size for training (default: 128)
        loss_mask: 'all' to compute loss on all tokens, 'last' to compute only on last token
        seed: Random seed
    Returns:
        Dictionary containing:
            train_losses: list of training losses
            test_losses: list of test losses
            eval_results: list of evaluation results at each evaluation step
            eval_steps: list of step numbers where evaluation was performed
            memory_stats: dictionary with memory usage statistics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses = []
    test_losses = []
    eval_results = []
    eval_steps = []

    # Setup activation hooks once (will be reused for all evaluations)
    hooks, activation_dict = setup_activation_hooks(model)

    # Initial memory stats
    initial_memory = print_gpu_memory_stats("Initial: ")

    # Get training data size
    num_train_samples = train_inputs.shape[0]

    for i in range(n_steps):

        if i == n_steps // 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        # Sample a random batch from training data
        # train_inputs is on CPU, so we sample indices and move to GPU
        batch_indices = torch.randint(0, num_train_samples, (batch_size,))
        batch_inputs = train_inputs[batch_indices].to(device)
        batch_targets = train_targets[batch_indices].to(device)

        # Training step
        noise = torch.randn_like(batch_inputs) * noise_scale
        if variable_dt:
            # Don't corrupt the dt query channel(s): noise only the position
            # channels [0:output_dim]; dt is an exact, known input, not an
            # observation. (For the standard case output_dim == input_dim, so
            # this branch is skipped and all channels are noised as before.)
            noise[..., model.config.output_dim:] = 0.0
        inputs_noised = batch_inputs + noise
        if rollout_horizon == 1:
            predictions, _ = model.forward(inputs_noised, None)  # Get predictions without loss
            train_loss = compute_loss_with_mask(predictions, batch_targets, loss_mask=loss_mask)
        else:
            # Rollout loss: free-run rollout_horizon steps from the noised window,
            # MSE against the rollout_horizon true future points (batch_targets is
            # (batch, rollout_horizon, 2)).
            predictions = rollout_predict(model, inputs_noised, rollout_horizon, model.config.block_size)
            train_loss = F.mse_loss(predictions, batch_targets)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(train_loss.item())

        # Clear intermediate variables to save memory
        del inputs_noised, predictions, batch_inputs, batch_targets, batch_indices

        # Compute test loss (no gradient) - use batch
        with torch.no_grad():
            test_batch_indices = torch.randint(0, test_inputs.shape[0], (batch_size,))
            test_batch_inputs = test_inputs[test_batch_indices].to(device)
            test_batch_targets = test_targets[test_batch_indices].to(device)
            if rollout_horizon == 1:
                test_predictions, _ = model.forward(test_batch_inputs, None)
                test_loss = compute_loss_with_mask(test_predictions, test_batch_targets, loss_mask=loss_mask)
            else:
                test_predictions = rollout_predict(model, test_batch_inputs, rollout_horizon, model.config.block_size)
                test_loss = F.mse_loss(test_predictions, test_batch_targets)
            del test_batch_inputs, test_batch_targets, test_batch_indices, test_predictions
            test_losses.append(test_loss.item())

        if i % 100 == 0:
            memory_stats = get_gpu_memory_stats()
            print(f"Step {i}, Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}, "
                    f"GPU Memory: {memory_stats['allocated_gb']:.3f} GB")
            # Periodically clear GPU cache to prevent memory fragmentation
            if i % 500 == 0 and i > 0:
                clear_gpu_cache()

        # Evaluate at specified frequency. Skipped entirely for variable_dt:
        # the trajectory-rollout error and the linear/geometry probes assume
        # 2-channel positions on a fixed-dt grid, neither of which holds here.
        # Per-step train/test losses are still logged; the variable-dt
        # mechanistic story comes from the downstream ELM dt-sweep analysis.
        if (not variable_dt) and (i % prob_freq == 0):
            print(f"\nEvaluating at step {i+1}...")
            eval_step_results = {}

            # PROBES DISABLED during training for speed - only run on final evaluation
            if i == n_steps - 1:
                # Only collect activations on final evaluation
                print("\nCollecting activations for final probes...")
                hooks, activation_dict = setup_activation_hooks(model)

                sample_size_activations = min(1000, train_inputs.shape[0])
                sample_indices = torch.randint(0, train_inputs.shape[0], (sample_size_activations,))

                # Map sequence indices to trajectory indices
                if train_inputs.shape[0] > train_trajectories.shape[0]:
                    num_sequences_per_trajectory = train_inputs.shape[0] // train_trajectories.shape[0]
                    trajectory_indices = (sample_indices.numpy() // num_sequences_per_trajectory)
                    trajectory_indices = np.clip(trajectory_indices, 0, train_trajectories.shape[0] - 1)
                else:
                    trajectory_indices = sample_indices.numpy()

                sample_trajectories = train_trajectories[trajectory_indices]
                sample_inputs = train_inputs[sample_indices].to(device)
                sample_targets = train_targets[sample_indices].to(device)

                predictions, gravitational_force = collect_activations(model, sample_inputs, sample_targets, activation_dict)
            else:
                # Skip probes during training
                activation_dict = {}
                gravitational_force = None
                trajectory_indices = None

            # Only compute probes on final evaluation
            if i == n_steps - 1 and gravitational_force is not None:
                print("\nRunning linear probes on final evaluation...")
                probe_results = run_linear_probes(activation_dict, gravitational_force)
                eval_step_results['probe_results'] = probe_results

                # Run geometry probes if orbital parameters are available
                if orbital_params is not None:
                    print("Running geometry probes on final evaluation...")
                    geometry_probe_results = run_geometry_probes(activation_dict, orbital_params, trajectory_indices)
                    eval_step_results['geometry_probe_results'] = geometry_probe_results
                else:
                    eval_step_results['geometry_probe_results'] = None
            else:
                # Skip probes during training for speed
                eval_step_results['probe_results'] = None
                eval_step_results['geometry_probe_results'] = None

            # Clear activations from GPU after probing to free memory
            for key in list(activation_dict.keys()):
                del activation_dict[key]
            activation_dict.clear()
            # Variables will be garbage collected automatically

            # Generate trajectory and compute error stats for train and test
            conditioning_length = 50
            block_size = model.config.block_size

            # Train error stats - use smaller sample and move to GPU only when needed
            # For evaluation, we need full trajectories, not chopped sequences
            train_sample_size = min(500, train_trajectories.shape[0])  # Reduced from 1000
            train_sample_indices = torch.randint(0, train_trajectories.shape[0], (train_sample_size,))
            train_sample_trajectories = train_trajectories[train_sample_indices.numpy()]  # Keep on CPU
            # Use full trajectories for evaluation (first conditioning_length points as input)
            train_sample_inputs = torch.from_numpy(train_sample_trajectories[:, :conditioning_length]).float().to(device)  # Move to GPU
            error_stats_train = generate_trajectory_and_compute_error(
                model, train_sample_inputs, train_sample_trajectories, conditioning_length, block_size=block_size
            )
            del train_sample_inputs, train_sample_indices

            # Test error stats - use smaller sample and move to GPU only when needed
            test_sample_size = min(500, test_trajectories.shape[0])  # Reduced from 1000
            test_sample_indices = torch.randint(0, test_trajectories.shape[0], (test_sample_size,))
            test_sample_trajectories = test_trajectories[test_sample_indices.numpy()]  # Keep on CPU
            # Use full trajectories for evaluation (first conditioning_length points as input)
            test_sample_inputs = torch.from_numpy(test_sample_trajectories[:, :conditioning_length]).float().to(device)  # Move to GPU
            error_stats_test = generate_trajectory_and_compute_error(
                model, test_sample_inputs, test_sample_trajectories, conditioning_length, block_size=block_size
            )
            del test_sample_inputs, test_sample_indices

            # Clear sample data from probing (only exists on final evaluation)
            if i == n_steps - 1:
                del sample_inputs, sample_targets, sample_trajectories, sample_indices
            clear_gpu_cache()
            eval_step_results['error_stats_train'] = error_stats_train
            eval_step_results['error_stats_test'] = error_stats_test
            eval_step_results['step'] = i + 1

            eval_results.append(eval_step_results)
            eval_steps.append(i + 1)

            # Clear GPU cache after evaluation
            clear_gpu_cache()

            memory_stats = print_gpu_memory_stats(f"After evaluation at step {i+1}: ")
            print(f"Evaluation at step {i+1} completed.\n")

    # Final memory stats
    final_memory = print_gpu_memory_stats("Final: ")
    peak_memory = get_gpu_memory_stats()

    for hook in hooks:
        hook.remove()
    activation_dict.clear()
    clear_gpu_cache()

    memory_stats = {
        'initial': initial_memory,
        'final': final_memory,
        'peak': peak_memory,
    }

    print(f"Final memory stats: {memory_stats}")

    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'eval_results': eval_results,
        'eval_steps': eval_steps,
        'memory_stats': memory_stats,
    }


def generate_trajectory_and_compute_error(model, inputs, trajectories, conditioning_length=50, block_size=None):
    """
    Generate trajectory continuation using the model for all trajectories.

    Args:
        model: trained model
        inputs: input trajectories (for conditioning)
        trajectories: full original trajectories (for comparison)
        conditioning_length: number of initial steps to use as conditioning
        block_size: block size of the model (needed if block_size < num_points_per_trajectory)

    Returns:
        error_stats: dictionary with aggregated error statistics across all trajectories
    """
    model.eval()

    num_trajectories = inputs.shape[0]
    num_points_per_trajectory = trajectories.shape[1]

    # If block_size < num_points_per_trajectory, we need to use sliding window generation
    if block_size is not None and block_size < num_points_per_trajectory:
        # Use sliding window approach: use last block_size points to predict next point
        num_steps_to_generate = num_points_per_trajectory - conditioning_length

        print(f"Evaluating on {num_trajectories} trajectories...")
        print(f"Conditioning length: {conditioning_length}, Generating {num_steps_to_generate} more steps per trajectory...")
        print(f"Using sliding window with block_size={block_size}")

        with torch.no_grad():
            # Use first conditioning_length steps as initial conditioning
            generated_seq = inputs[:, :conditioning_length].clone()  # (num_trajectories, conditioning_length, 2)

            # Generate one point at a time using the last block_size points
            for step in range(num_steps_to_generate):
                # Use the last block_size points as input
                current_input = generated_seq[:, -block_size:, :]  # (num_trajectories, block_size, 2)

                # Generate next point
                predictions, _ = model.forward(current_input, None)
                next_point = predictions[:, -1, :].unsqueeze(1)  # (num_trajectories, 1, 2)

                # Append to generated sequence
                generated_seq = torch.cat([generated_seq, next_point], dim=1)

            # Convert to numpy (move to CPU if on GPU)
            generated_np = generated_seq.cpu().numpy()  # (num_trajectories, total_length, 2)
            true_trajectories_np = trajectories  # (num_trajectories, num_points_per_trajectory, 2)

            # Clear GPU tensor
            del generated_seq

            # Extract only the generated portion (exclude conditioning_length)
            generated_only = generated_np[:, conditioning_length:, :]  # (num_trajectories, num_steps_to_generate, 2)

            # Extract the corresponding true trajectory portion (also exclude conditioning_length)
            true_generated = true_trajectories_np[:, conditioning_length:, :]  # (num_trajectories, num_steps_to_generate, 2)

            # Calculate errors only on the generated portion (not conditioning)
            # Align sequences in case of length mismatch
            min_len = min(generated_only.shape[1], true_generated.shape[1])
            generated_aligned = generated_only[:, :min_len, :]  # (num_trajectories, min_len, 2)
            true_aligned = true_generated[:, :min_len, :]  # (num_trajectories, min_len, 2)

            # Compute position errors for all trajectories
            position_errors = np.sqrt(np.sum((generated_aligned - true_aligned)**2, axis=2))  # (num_trajectories, min_len)
            all_position_errors = position_errors.flatten()  # Flatten to get all errors
            # Compute mean across trajectories before deleting
            position_errors_mean = np.mean(position_errors, axis=0)  # (min_len,)

            # Compute R² scores for each trajectory
            all_r2_x = []
            all_r2_y = []
            for traj_idx in range(num_trajectories):
                r2_x = r2_score(true_aligned[traj_idx, :, 0], generated_aligned[traj_idx, :, 0])
                r2_y = r2_score(true_aligned[traj_idx, :, 1], generated_aligned[traj_idx, :, 1])
                all_r2_x.append(r2_x)
                all_r2_y.append(r2_y)

            # Clear intermediate variables
            del generated_np, true_trajectories_np, generated_only, true_generated, generated_aligned, true_aligned, position_errors
            clear_gpu_cache()
    else:
        # Standard generation (block_size >= num_points_per_trajectory)
        num_steps_to_generate = num_points_per_trajectory - conditioning_length

        print(f"Evaluating on {num_trajectories} trajectories...")
        print(f"Conditioning length: {conditioning_length}, Generating {num_steps_to_generate} more steps per trajectory...")

        with torch.no_grad():
            # Use first N steps as conditioning for all trajectories (batch processing)
            conditioning_seq = inputs[:, :conditioning_length]  # shape: (num_trajectories, conditioning_length, 2)

            # Generate remaining steps for all trajectories in batch
            generated_seq = model.generate(conditioning_seq, max_new_tokens=num_steps_to_generate)
            # generated_seq shape: (num_trajectories, conditioning_length + num_steps_to_generate, 2)

            # Convert to numpy (move to CPU if on GPU)
            generated_np = generated_seq.cpu().numpy()  # (num_trajectories, total_length, 2)
            true_trajectories_np = trajectories  # (num_trajectories, num_points_per_trajectory, 2)

            # Clear GPU tensor
            del generated_seq, conditioning_seq

            # Extract only the generated portion (exclude conditioning_length)
            # The generated_seq includes conditioning + generated, so we take only the generated part
            generated_only = generated_np[:, conditioning_length:, :]  # (num_trajectories, num_steps_to_generate, 2)

            # Extract the corresponding true trajectory portion (also exclude conditioning_length)
            true_generated = true_trajectories_np[:, conditioning_length:, :]  # (num_trajectories, num_steps_to_generate, 2)

            # Calculate errors only on the generated portion (not conditioning)
            # Align sequences in case of length mismatch
            min_len = min(generated_only.shape[1], true_generated.shape[1])
            generated_aligned = generated_only[:, :min_len, :]  # (num_trajectories, min_len, 2)
            true_aligned = true_generated[:, :min_len, :]  # (num_trajectories, min_len, 2)

            # Compute position errors for all trajectories
            position_errors = np.sqrt(np.sum((generated_aligned - true_aligned)**2, axis=2))  # (num_trajectories, min_len)
            all_position_errors = position_errors.flatten()  # Flatten to get all errors
            # Compute mean across trajectories before deleting
            position_errors_mean = np.mean(position_errors, axis=0)  # (min_len,)

            # Compute R² scores for each trajectory
            all_r2_x = []
            all_r2_y = []
            for traj_idx in range(num_trajectories):
                r2_x = r2_score(true_aligned[traj_idx, :, 0], generated_aligned[traj_idx, :, 0])
                r2_y = r2_score(true_aligned[traj_idx, :, 1], generated_aligned[traj_idx, :, 1])
                all_r2_x.append(r2_x)
                all_r2_y.append(r2_y)

            # Clear intermediate variables
            del generated_np, true_trajectories_np, generated_only, true_generated, generated_aligned, true_aligned, position_errors
            clear_gpu_cache()

    # Aggregate statistics
    # position_errors_mean was computed in the branch above
    mean_error = np.mean(all_position_errors)
    max_error = np.max(all_position_errors)
    std_error = np.std(all_position_errors)

    mean_r2_x = np.mean(all_r2_x)
    mean_r2_y = np.mean(all_r2_y)
    std_r2_x = np.std(all_r2_x)
    std_r2_y = np.std(all_r2_y)

    error_stats = {
        'position_errors': position_errors_mean,
        'mean_error': mean_error,
        'std_error': std_error,
        'max_error': max_error,
        'mean_r2_x': mean_r2_x,
        'std_r2_x': std_r2_x,
        'mean_r2_y': mean_r2_y,
        'std_r2_y': std_r2_y,
        'all_r2_x': all_r2_x,
        'all_r2_y': all_r2_y,
    }

    print(f"\nError Statistics (aggregated across all {num_trajectories} trajectories):")
    print(f"Mean position error: {mean_error:.6f} ± {std_error:.6f}")
    print(f"Max position error: {max_error:.6f}")
    print(f"R² scores:")
    print(f"  X coordinate: {mean_r2_x:.6f} ± {std_r2_x:.6f}")
    print(f"  Y coordinate: {mean_r2_y:.6f} ± {std_r2_y:.6f}")

    return error_stats


def train_one_model(block_size=100, noise_scale=0.1, lr=1e-3, n_layer=2, n_embd=16, num_trajectories=10000,
                    n_steps=1001, prob_freq=100, loss_mask='all', seed=1, batch_size=128, rollout_horizon=1,
                    variable_dt=False):
    """
    Train a single model with specified hyperparameters.

    Args:
        block_size: Block size for the model (if < num_points_per_trajectory, trajectories will be chopped)
        noise_scale: Scale of noise added during training
        lr: Learning rate
        n_layer: Number of transformer layers
        n_embd: Embedding dimension
        num_trajectories: Number of trajectories to generate
        loss_mask: 'all' to compute loss on all tokens, 'last' to compute only on last token

    Returns:
        Dictionary containing model results and statistics
    """
    print(f"\n{'='*80}")
    print(f"Training model: block_size={block_size}, noise_scale={noise_scale}, lr={lr}, n_layer={n_layer}, n_embd={n_embd}, num_trajectories={num_trajectories}, loss_mask={loss_mask}")
    print(f"{'='*80}")

    # Ensure num_trajectories is an integer
    num_trajectories = int(num_trajectories)
    if num_trajectories <= 0:
        raise ValueError(f"num_trajectories must be a positive integer, got {num_trajectories}")

    # Variable-dt runs read the 3-channel (x, y, dt) dataset and predict only the
    # first `target_dim` channels (the next (x, y)). Standard runs are unchanged.
    data_dir = 'data_cv_vardt' if variable_dt else 'data_cv'
    target_dim = 2 if variable_dt else None
    if variable_dt and rollout_horizon > 1:
        raise NotImplementedError(
            "variable_dt currently supports rollout_horizon=1 (single-step) only; "
            "feeding a 2-channel prediction back as a 3-channel (x,y,dt) token needs "
            "a dt query schedule (see model.generate(dt_next=...))."
        )

    # Load trajectories. Only load what we need (2*num_trajectories for train+test).
    print(f"Loading Kepler orbit trajectories from {data_dir}...")
    trajectories = load_trajectories(data_dir, num_trajectories_needed=2*num_trajectories)
    print(f"Trajectories shape: {trajectories.shape}")

    if trajectories.shape[0] == 0:
        raise ValueError(f"No trajectories loaded. Please check that the data files exist in data_cv/")

    print(f"Position range: x=[{trajectories[:,:,0].min():.3f}, {trajectories[:,:,0].max():.3f}], "
          f"y=[{trajectories[:,:,1].min():.3f}, {trajectories[:,:,1].max():.3f}]")

    # Split into train and test sets (50/50 split)
    num_traj = trajectories.shape[0]
    train_trajectories = trajectories[:num_traj//2]
    test_trajectories = trajectories[num_traj//2:]

    # Save sizes before deleting trajectories
    train_size = train_trajectories.shape[0]
    test_size = test_trajectories.shape[0]

    # Clear trajectories to save memory (we'll keep train_trajectories and test_trajectories for evaluation)
    # But we need to keep them for evaluation, so don't delete yet

    print(f"\nSplit: {train_size} training trajectories, {test_size} test trajectories")

    # Load orbital parameters
    print(f"Loading orbital parameters from {data_dir}...")
    orbital_params = load_orbital_params(data_dir, num_trajectories_needed=2*num_trajectories)
    if orbital_params:
        print(f"Loaded {len(orbital_params)} orbital parameters")
        # Split orbital parameters to match train/test split
        train_orbital_params = orbital_params[:num_traj//2]
        test_orbital_params = orbital_params[num_traj//2:]
        # For probing, we'll use train orbital params (since we probe on training data)
        orbital_params_for_probing = train_orbital_params
    else:
        print("Warning: No orbital parameters found. Geometry probes will be skipped.")
        orbital_params_for_probing = None

    # Prepare inputs and targets based on block_size
    if block_size < num_points_per_trajectory:
        # Chop trajectories into sequences of length block_size + target_len
        print(f"\nBlock size ({block_size}) < num_points_per_trajectory ({num_points_per_trajectory})")
        print("Chopping trajectories into sequences...")
        num_sequences_per_traj = num_points_per_trajectory // block_size
        print(f"Randomly selecting {num_sequences_per_traj} sequences per trajectory (instead of all sliding windows)")
        if rollout_horizon > 1:
            print(f"Rollout loss enabled: horizon={rollout_horizon} (targets are {rollout_horizon} future points)")
        train_inputs_np, train_targets_np = chop_trajectories_into_sequences(train_trajectories, block_size, seed=seed, target_len=rollout_horizon, target_dim=target_dim)
        test_inputs_np, test_targets_np = chop_trajectories_into_sequences(test_trajectories, block_size, seed=seed+1, target_len=rollout_horizon, target_dim=target_dim)  # Different seed for test
        print(f"Train sequences: {train_inputs_np.shape[0]}, Test sequences: {test_inputs_np.shape[0]}")
    elif rollout_horizon > 1:
        raise NotImplementedError(
            "rollout_horizon > 1 is only supported when block_size < num_points_per_trajectory "
            "(the chopped-window regime, e.g. block_size=2)."
        )
    else:
        # Standard approach: use full trajectories
        print(f"\nBlock size ({block_size}) >= num_points_per_trajectory ({num_points_per_trajectory})")
        print("Using full trajectories...")
        train_inputs_np = train_trajectories[:,:-1]  # shape: (train_size, num_points-1, C)
        train_targets_np = train_trajectories[:,1:]   # shape: (train_size, num_points-1, C)
        test_inputs_np = test_trajectories[:,:-1]  # shape: (test_size, num_points-1, C)
        test_targets_np = test_trajectories[:,1:]   # shape: (test_size, num_points-1, C)
        if target_dim is not None:
            # Predict only the first target_dim channels (next (x, y), not dt).
            train_targets_np = train_targets_np[..., :target_dim]
            test_targets_np = test_targets_np[..., :target_dim]

    # Convert to PyTorch tensors - keep on CPU to save GPU memory, will move batches to GPU during training
    train_inputs = torch.from_numpy(train_inputs_np).float()  # keep on CPU
    train_targets = torch.from_numpy(train_targets_np).float()   # keep on CPU
    test_inputs = torch.from_numpy(test_inputs_np).float()  # keep on CPU
    test_targets = torch.from_numpy(test_targets_np).float()   # keep on CPU

    print(f"\nTrain input shape: {train_inputs.shape}, Train target shape: {train_targets.shape}")
    print(f"Test input shape: {test_inputs.shape}, Test target shape: {test_targets.shape}")

    # Clear numpy arrays to save memory
    del train_inputs_np, train_targets_np, test_inputs_np, test_targets_np

    # Setup model
    print("\nSetting up model...")
    model = setup_model(block_size=block_size, n_layer=n_layer, n_embd=n_embd, device=device, variable_dt=variable_dt)
    print_gpu_memory_stats("After model setup: ")

    # Initial forward pass - use smaller batch to save memory
    with torch.no_grad():
        init_batch_size = min(128, train_inputs.shape[0])
        init_train_indices = torch.randint(0, train_inputs.shape[0], (init_batch_size,))
        init_test_indices = torch.randint(0, test_inputs.shape[0], (init_batch_size,))

        init_train_inputs = train_inputs[init_train_indices].to(device)
        init_train_targets = train_targets[init_train_indices].to(device)
        init_test_inputs = test_inputs[init_test_indices].to(device)
        init_test_targets = test_targets[init_test_indices].to(device)

        if rollout_horizon == 1:
            train_predictions, _ = model.forward(init_train_inputs, None)
            train_loss = compute_loss_with_mask(train_predictions, init_train_targets, loss_mask=loss_mask)
            test_predictions, _ = model.forward(init_test_inputs, None)
            test_loss = compute_loss_with_mask(test_predictions, init_test_targets, loss_mask=loss_mask)
        else:
            train_predictions = rollout_predict(model, init_train_inputs, rollout_horizon, model.config.block_size)
            train_loss = F.mse_loss(train_predictions, init_train_targets)
            test_predictions = rollout_predict(model, init_test_inputs, rollout_horizon, model.config.block_size)
            test_loss = F.mse_loss(test_predictions, init_test_targets)

        print(f"Initial train loss: {train_loss.item():.6f}, Initial test loss: {test_loss.item():.6f}")
        print(f"Train predictions shape: {train_predictions.shape}")

        del train_predictions, test_predictions
        del init_train_inputs, init_train_targets, init_test_inputs, init_test_targets
        del init_train_indices, init_test_indices

    print_gpu_memory_stats("After initial forward pass: ")

    batch_size = batch_size * 100 // block_size

    # Training with periodic evaluation
    print("\nTraining model with periodic evaluation...")
    training_results = train_model(
        model, train_inputs, train_targets, test_inputs, test_targets, train_trajectories, test_trajectories,
        n_steps=n_steps, lr=lr, weight_decay=0.0, noise_scale=noise_scale, prob_freq=prob_freq,
        loss_mask=loss_mask, batch_size=batch_size, seed=seed, orbital_params=orbital_params_for_probing,
        rollout_horizon=rollout_horizon, variable_dt=variable_dt
    )

    train_losses = training_results['train_losses']
    test_losses = training_results['test_losses']
    eval_results = training_results['eval_results']
    eval_steps = training_results['eval_steps']
    memory_stats = training_results.get('memory_stats', {})

    print("\nTraining completed!")
    print_gpu_memory_stats("After training: ")

    # Print memory summary
    if torch.cuda.is_available() and memory_stats:
        print(f"\nMemory Summary:")
        if 'initial' in memory_stats:
            print(f"  Initial: {memory_stats['initial']['allocated_gb']:.3f} GB")
        if 'final' in memory_stats:
            print(f"  Final: {memory_stats['final']['allocated_gb']:.3f} GB")
        if 'peak' in memory_stats:
            print(f"  Peak: {memory_stats['peak']['max_allocated_gb']:.3f} GB")
        if 'initial' in memory_stats and 'final' in memory_stats:
            print(f"  Memory increase: {memory_stats['final']['allocated_gb'] - memory_stats['initial']['allocated_gb']:.3f} GB")

    # Get final evaluation results (last evaluation)
    final_eval = eval_results[-1] if eval_results else None

    # Clear large tensors (they're on CPU, but still free memory)
    # NOTE: Don't delete model - we need it for checkpoint saving!
    del train_inputs, train_targets, test_inputs, test_targets
    del train_trajectories, test_trajectories
    clear_gpu_cache()

    # Return results
    results = {
        'block_size': block_size,
        'noise_scale': noise_scale,
        'lr': lr,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'num_trajectories': num_traj,
        'train_size': train_size,
        'test_size': test_size,
        'loss_mask': loss_mask,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_test_loss': test_losses[-1] if test_losses else None,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'eval_results': eval_results,
        'eval_steps': eval_steps,
        'final_error_stats_train': final_eval['error_stats_train'] if final_eval and 'error_stats_train' in final_eval else None,
        'final_error_stats_test': final_eval['error_stats_test'] if final_eval and 'error_stats_test' in final_eval else None,
        'final_probe_results': final_eval['probe_results'] if final_eval else None,
        'final_geometry_probe_results': final_eval['geometry_probe_results'] if final_eval and 'geometry_probe_results' in final_eval else None,
        'memory_stats': memory_stats,
    }

    return results, model


def sweep_parameters(block_size_list, num_trajectories_list, noise_scale_list, loss_mask_list=['all'],
                     lr=1e-3, n_layer=2, n_embd=32, n_steps=1001, prob_freq=100, seed=1, rollout_horizon=1,
                     variable_dt=False):
    """
    Sweep over block_size, num_trajectories, noise_scale, and loss_mask parameters.

    Args:
        block_size_list: List of block_size values to sweep
        num_trajectories_list: List of num_trajectories values to sweep
        noise_scale_list: List of noise_scale values to sweep
        loss_mask_list: List of loss_mask values to sweep (default: ['all'])
        lr: Learning rate (fixed)
        n_layer: Number of transformer layers (fixed)
        n_embd: Embedding dimension (fixed)

    Returns:
        Dictionary mapping (block_size, num_trajectories, noise_scale, loss_mask) to results
    """

    print(f"\n{'='*80}")
    print(f"Starting parameter sweep:")
    print(f"  block_size: {block_size_list}")
    print(f"  num_trajectories: {num_trajectories_list}")
    print(f"  noise_scale: {noise_scale_list}")
    print(f"  loss_mask: {loss_mask_list}")
    print(f"  lr: {lr} (fixed)")
    print(f"  n_layer: {n_layer} (fixed)")
    print(f"  n_embd: {n_embd} (fixed)")
    print(f"{'='*80}\n")

    total_runs = len(block_size_list) * len(num_trajectories_list) * len(noise_scale_list) * len(loss_mask_list)
    run_count = 0

    # Tag filenames when using a rollout-loss horizon > 1 so the checkpoint/results
    # do not clobber the standard single-step (horizon=1) runs.
    rollout_tag = '' if rollout_horizon == 1 else f'_rollout_{rollout_horizon}'
    # Tag variable-dt runs so they sit alongside (not on top of) the fixed-dt ones.
    vardt_tag = '_vardt' if variable_dt else ''

    num_traj = 10000

    # Pre-filter configurations to skip existing results
    configs_to_run = []
    for block_size in block_size_list:
        for noise_scale in noise_scale_list:
            for num_traj in num_trajectories_list:
                for loss_mask in loss_mask_list:
                    results_filename = f'./results/kepler_cv_blocksize_elm/results_block_size_{block_size}_num_trajectories_{num_traj}_noise_scale_{noise_scale}_loss_mask_{loss_mask}_seed_{seed}{rollout_tag}{vardt_tag}.pt'
                    if not os.path.exists(results_filename):
                        configs_to_run.append((block_size, num_traj, noise_scale, loss_mask))
                    else:
                        print(f"Skipping existing result: {results_filename}")

    print(f"\nTotal configurations: {total_runs}, Already completed: {total_runs - len(configs_to_run)}, To run: {len(configs_to_run)}")

    for block_size, num_traj, noise_scale, loss_mask in configs_to_run:
        run_count += 1
        print(f"\n[{run_count}/{len(configs_to_run)}] Running: block_size={block_size}, num_trajectories={num_traj}, noise_scale={noise_scale}, loss_mask={loss_mask}")
        results, model = train_one_model(
            block_size=block_size,
            noise_scale=noise_scale,
            lr=lr,
            n_layer=n_layer,
            n_embd=n_embd,
            num_trajectories=num_traj,
            n_steps=n_steps,
            prob_freq=prob_freq,
            loss_mask=loss_mask,
            seed=seed,
            rollout_horizon=rollout_horizon,
            variable_dt=variable_dt
        )

        # save results to file (separate directory for ELM variant)
        results_filename = f'./results/kepler_cv_blocksize_elm/results_block_size_{block_size}_num_trajectories_{num_traj}_noise_scale_{noise_scale}_loss_mask_{loss_mask}{rollout_tag}{vardt_tag}.pt'
        os.makedirs(os.path.dirname(results_filename), exist_ok=True)
        torch.save(results, results_filename)
        print(f"Saved results to {results_filename}")

        # save model checkpoint for ELM analysis
        checkpoint_filename = f'./checkpoints/kepler_cv_blocksize_elm/model_block_size_{block_size}_num_trajectories_{num_traj}_noise_scale_{noise_scale}{rollout_tag}{vardt_tag}.pt'
        os.makedirs(os.path.dirname(checkpoint_filename), exist_ok=True)
        model.save_checkpoint(
            checkpoint_filename,
            step=n_steps,
            extra={
                'block_size': block_size,
                'num_trajectories': num_traj,
                'noise_scale': noise_scale,
                'rollout_horizon': rollout_horizon,
                'final_train_loss': results['final_train_loss'],
                'final_test_loss': results['final_test_loss'],
            }
        )
        print(f"Saved checkpoint to {checkpoint_filename}")


    print(f"\n{'='*80}")
    print(f"Parameter sweep completed! Total runs: {len(configs_to_run)}")
    print(f"{'='*80}\n")

    return None


def main():
    """Main execution function."""
    # Run a single model with default parameters
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_trajectories_list = [10000]
    #block_size_list = [2, 100]  # Quick test: just endpoints
    #block_size_list = [60, 70, 80, 90]
    block_size_list = [2, 100]
    noise_scale_list = [0.1]
    loss_mask_list = ['all']
    n_steps = 20000
    prob_freq = 100
    sweep_parameters(block_size_list, num_trajectories_list, noise_scale_list, loss_mask_list,
                     n_steps=n_steps, prob_freq=prob_freq, seed=seed)


if __name__ == "__main__":
    main()
