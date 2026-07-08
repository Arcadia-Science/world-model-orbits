"""Kepler trajectory dataset loading and windowing.

Loaders for the pre-generated `data_cv/` dataset (trajectories + orbital
parameters) and the routine that chops trajectories into fixed-length training
windows. See `data_generation` for how the dataset is produced.
"""

# ---------------------------------------------------------------------------
# Adapted from Liu et al., "From Kepler to Newton: Inductive Biases Guide
# Learned World Models in Transformers" (arXiv:2602.06923), CC BY 4.0
# (https://creativecommons.org/licenses/by/4.0/).
# Source: https://github.com/KindXiaoming/newton-kepler
# Modification: trajectory loading/windowing extracted from the upstream
# training script. Full attribution in NOTICE.md.
# ---------------------------------------------------------------------------

import os
import importlib.util
import numpy as np
import torch


def load_trajectories(data_dir='data_cv', num_trajectories_needed=None):
    """
    Load continuous trajectories from the data_cv folder.
    Supports both chunked format (new) and single file format (old, for backward compatibility).

    Args:
        data_dir: Directory containing the saved trajectories
        num_trajectories_needed: Number of trajectories to load (None = load all available)

    Returns:
        trajectories: array of shape (num_trajectories, num_points, 2)
    """
    # Check for chunked format first
    metadata_path = os.path.join(data_dir, 'metadata.pt')
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, weights_only=False)
        chunk_size = metadata.get('chunk_size', 10000)
        num_chunks = metadata.get('num_chunks', 0)
        total_available = metadata.get('num_trajectories', 0)

        # Determine how many trajectories to load
        if num_trajectories_needed is None:
            num_trajectories_needed = total_available
        else:
            num_trajectories_needed = min(num_trajectories_needed, total_available)

        # Ensure num_trajectories_needed is an integer
        num_trajectories_needed = int(num_trajectories_needed)
        chunk_size = int(chunk_size)
        num_chunks = int(num_chunks)

        # Only try to load chunks if we have chunks available
        if num_chunks > 0 and num_trajectories_needed > 0:
            print(f"Loading {num_trajectories_needed:,} trajectories from {num_chunks} chunks...")

            # Calculate which chunks we need
            num_chunks_needed = (num_trajectories_needed + chunk_size - 1) // chunk_size
            num_chunks_needed = min(num_chunks_needed, num_chunks)
            num_chunks_needed = int(num_chunks_needed)

            # Load chunks
            chunks = []
            trajectories_loaded = 0

            for chunk_idx in range(num_chunks_needed):
                chunk_filename = os.path.join(data_dir, f'trajectories_chunk_{chunk_idx:06d}.pt')
                if os.path.exists(chunk_filename):
                    chunk_data = torch.load(chunk_filename, weights_only=False)
                    if isinstance(chunk_data, torch.Tensor):
                        chunk_data = chunk_data.numpy()

                    remaining_needed = num_trajectories_needed - trajectories_loaded
                    if chunk_data.shape[0] <= remaining_needed:
                        chunks.append(chunk_data)
                        trajectories_loaded += chunk_data.shape[0]
                    else:
                        # Only take what we need
                        chunks.append(chunk_data[:remaining_needed])
                        trajectories_loaded += remaining_needed
                        break
                else:
                    print(f"Warning: Chunk {chunk_idx} not found. Falling back to old format.")
                    chunks = []  # Clear chunks to trigger fallback
                    break

            # Concatenate chunks if we successfully loaded any
            if chunks:
                trajectories = np.concatenate(chunks, axis=0)
                # Ensure we have exactly the number needed
                if trajectories.shape[0] > num_trajectories_needed:
                    trajectories = trajectories[:num_trajectories_needed]
                print(f"Loaded {trajectories.shape[0]:,} trajectories from chunks")
                return trajectories

        # If chunks weren't available or loading failed, fall through to old format
        print(f"Chunked format not available or incomplete, falling back to old format...")

    # Fallback to old format (single file) for backward compatibility
    pt_path = os.path.join(data_dir, 'trajectories.pt')
    npy_path = os.path.join(data_dir, 'trajectories.npy')

    if os.path.exists(pt_path):
        trajectories = torch.load(pt_path, weights_only=False)
        if isinstance(trajectories, torch.Tensor):
            trajectories = trajectories.numpy()

        # Limit to num_trajectories_needed if specified
        if num_trajectories_needed is not None and trajectories.shape[0] > num_trajectories_needed:
            trajectories = trajectories[:num_trajectories_needed]

        print(f"Loaded {trajectories.shape[0]:,} trajectories from {pt_path}")
        return trajectories
    elif os.path.exists(npy_path):
        trajectories = np.load(npy_path)

        # Limit to num_trajectories_needed if specified
        if num_trajectories_needed is not None and trajectories.shape[0] > num_trajectories_needed:
            trajectories = trajectories[:num_trajectories_needed]

        print(f"Loaded {trajectories.shape[0]:,} trajectories from {npy_path}")
        return trajectories
    else:
        raise FileNotFoundError(
            f"Trajectories not found in {data_dir}. "
            f"Please run generate_kepler_cv.py first to generate the dataset."
        )


def chop_trajectories_into_sequences(trajectories, block_size, seed=None, target_len=1, target_dim=None):
    """
    Chop trajectories into sequences of length block_size + 1 when block_size < num_points_per_trajectory.

    Randomly selects num_points_per_trajectory // block_size sequences per trajectory to avoid
    creating too large a dataset. For example:
    - block_size=50: randomly select 2 pieces (100/50) of length 51
    - block_size=10: randomly select 10 pieces (100/10) of length 11

    Args:
        trajectories: array of shape (num_trajectories, num_points, C)
        block_size: block size for the model
        seed: random seed for reproducibility (optional)
        target_dim: if set, keep only the first `target_dim` channels of the
            target (the input keeps all C channels). Used by the variable-dt
            task where the input token is (x, y, dt) but the target is the next
            (x, y) only -- we predict position, never the dt query. None keeps
            all channels (standard coordinate->coordinate case).

    Returns:
        inputs: array of shape (num_sequences, block_size, C)
        targets: array of shape (num_sequences, target_len, target_dim or C)
    """
    if seed is not None:
        np.random.seed(seed)

    num_trajectories, num_points, _ = trajectories.shape
    # block_size input points + target_len future target points (target_len=1 is
    # the standard single-next-step objective; target_len=k provides the k future
    # ground-truth points needed for a rollout-loss horizon of k).
    seq_length = block_size + target_len
    num_sequences_per_trajectory = num_points // block_size  # e.g., 100 // 50 = 2, 100 // 10 = 10

    all_inputs = []
    all_targets = []

    for traj in trajectories:
        # Calculate valid starting positions (must have at least seq_length points remaining)
        max_start = num_points - seq_length + 1

        # Randomly select num_sequences_per_trajectory starting positions
        if max_start <= num_sequences_per_trajectory:
            # If we can't select enough unique positions, use all available
            start_positions = np.arange(max_start)
        else:
            # Randomly sample without replacement
            start_positions = np.random.choice(max_start, size=num_sequences_per_trajectory, replace=False)

        # Create sequences starting at selected positions
        for i in start_positions:
            input_seq = traj[i:i+block_size]  # (block_size, C) - all input channels
            target_seq = traj[i+block_size:i+block_size+target_len]  # (target_len, C)
            if target_dim is not None:
                target_seq = target_seq[:, :target_dim]  # predict first target_dim channels only (e.g. (x,y), not dt)
            all_inputs.append(input_seq)
            all_targets.append(target_seq)

    inputs = np.array(all_inputs)  # (num_sequences, block_size, 2)
    targets = np.array(all_targets)  # (num_sequences, 1, 2)

    return inputs, targets


def load_orbital_params(data_dir='data_cv', num_trajectories_needed=None):
    """
    Load orbital parameters from the data_cv folder.
    Supports chunked format (orbital_params_chunk_*.py files).

    Args:
        data_dir: Directory containing the saved orbital parameters
        num_trajectories_needed: Number of trajectories to load (None = load all available)

    Returns:
        orbital_params: list of dicts with keys: e, a, b, c, average_radius
    """
    # Check for chunked format
    metadata_path = os.path.join(data_dir, 'metadata.pt')
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, weights_only=False)
        chunk_size = metadata.get('chunk_size', 10000)
        num_chunks = metadata.get('num_chunks', 0)
        total_available = metadata.get('num_trajectories', 0)

        # Determine how many trajectories to load
        if num_trajectories_needed is None:
            num_trajectories_needed = total_available
        else:
            num_trajectories_needed = min(num_trajectories_needed, total_available)

        # Ensure num_trajectories_needed is an integer
        num_trajectories_needed = int(num_trajectories_needed)
        chunk_size = int(chunk_size)
        num_chunks = int(num_chunks)

        if num_chunks > 0 and num_trajectories_needed > 0:
            print(f"Loading {num_trajectories_needed:,} orbital parameters from {num_chunks} chunks...")

            # Calculate which chunks we need
            num_chunks_needed = (num_trajectories_needed + chunk_size - 1) // chunk_size
            num_chunks_needed = min(num_chunks_needed, num_chunks)
            num_chunks_needed = int(num_chunks_needed)

            # Load chunks
            all_orbital_params = []
            trajectories_loaded = 0

            for chunk_idx in range(num_chunks_needed):
                chunk_filename = os.path.join(data_dir, f'orbital_params_chunk_{chunk_idx:06d}.py')
                if os.path.exists(chunk_filename):
                    # Import the orbital_params from the Python file
                    spec = importlib.util.spec_from_file_location(f"orbital_params_chunk_{chunk_idx}", chunk_filename)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    chunk_data = module.orbital_params

                    remaining_needed = num_trajectories_needed - trajectories_loaded
                    if len(chunk_data) <= remaining_needed:
                        all_orbital_params.extend(chunk_data)
                        trajectories_loaded += len(chunk_data)
                    else:
                        # Only take what we need
                        all_orbital_params.extend(chunk_data[:remaining_needed])
                        trajectories_loaded += remaining_needed
                        break
                else:
                    print(f"Warning: Orbital parameters chunk {chunk_idx} not found.")
                    break

            if all_orbital_params:
                print(f"Loaded {len(all_orbital_params):,} orbital parameters from chunks")
                return all_orbital_params

    # If chunks weren't available, return empty list
    print(f"Orbital parameters not found in {data_dir}")
    return []
