"""Linear-probing suite (Liu et al. Fig 6 reproduction).

Registers activation hooks, collects hidden states, and regresses Newtonian
(force) and Keplerian (orbital-geometry) target variables against them to
measure what each model linearly encodes.
"""

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def setup_activation_hooks(model):
    """
    Set up hooks to collect intermediate activations from the model.

    For each transformer block, captures:
    1. Attention output before/after merging into residual stream
    2. MLP output before/after merging into residual stream
    3. MLP hidden activation after silu

    Returns:
        hooks: list of registered hooks
        activation_dict: dictionary to store activations
    """
    activation_dict = {}
    hooks = []
    n_layer = model.config.n_layer


    # Helper function to extract tensor from input/output
    def extract_tensor(x):
        if isinstance(x, tuple):
            return x[0].detach()
        return x.detach()

    # Hook for attention output BEFORE merging into residual
    def make_attn_output_hook(block_idx):
        def hook(module, input, output):
            activation_dict[f'block_{block_idx}_attn_output'] = extract_tensor(output)
        return hook

    # Hook for residual stream AFTER attention merge (input to MLP)
    def make_after_attn_merge_hook(block_idx):
        def hook(module, input):
            # MLP receives the residual stream after attention merge
            activation_dict[f'block_{block_idx}_after_attn_merge'] = extract_tensor(input)
        return hook

    # Hook for MLP output BEFORE merging into residual
    def make_mlp_output_hook(block_idx):
        def hook(module, input, output):
            activation_dict[f'block_{block_idx}_mlp_output'] = extract_tensor(output)
        return hook

    # Hook for residual stream AFTER MLP merge (output of block)
    def make_after_mlp_merge_hook(block_idx):
        def hook(module, input, output):
            # Block output is the residual stream after MLP merge
            activation_dict[f'block_{block_idx}_after_mlp_merge'] = extract_tensor(output)
        return hook

    # Hook for MLP hidden activation after silu
    def make_mlp_hidden_hook(block_idx):
        def hook(module, input, output):
            activation_dict[f'block_{block_idx}_mlp_hidden'] = extract_tensor(output)
        return hook

    # Register hooks for each transformer block
    for block_idx in range(n_layer):
        block = model.transformer.h[block_idx]

        # 1. Attention output before merge
        hook = block.attn.register_forward_hook(make_attn_output_hook(block_idx))
        hooks.append(hook)

        # 2. Residual after attention merge (input to MLP)
        hook = block.mlp.register_forward_pre_hook(make_after_attn_merge_hook(block_idx))
        hooks.append(hook)

        # 3. MLP output before merge
        hook = block.mlp.register_forward_hook(make_mlp_output_hook(block_idx))
        hooks.append(hook)

        # 4. Residual after MLP merge (block output)
        hook = block.register_forward_hook(make_after_mlp_merge_hook(block_idx))
        hooks.append(hook)

        # 5. MLP hidden activation after silu
        hook = block.mlp.silu.register_forward_hook(make_mlp_hidden_hook(block_idx))
        hooks.append(hook)

    # Also register hooks for input embedding and final layer norm
    def make_input_embed_hook():
        def hook(module, input, output):
            activation_dict['input_embed'] = extract_tensor(output)
        return hook

    def make_after_pos_emb_hook():
        def hook(module, input, output):
            activation_dict['after_pos_emb'] = extract_tensor(output)
        return hook

    def make_after_ln_f_hook():
        def hook(module, input, output):
            activation_dict['after_ln_f'] = extract_tensor(output)
        return hook

    hook = model.input_embedding.register_forward_hook(make_input_embed_hook())
    hooks.append(hook)

    hook = model.transformer.drop.register_forward_hook(make_after_pos_emb_hook())
    hooks.append(hook)

    hook = model.transformer.ln_f.register_forward_hook(make_after_ln_f_hook())
    hooks.append(hook)

    print(f"Hooks registered for {n_layer} transformer blocks:")
    print(f"  For each block: attn_output, after_attn_merge, mlp_output, after_mlp_merge, mlp_hidden")
    print(f"  Additional: input_embed, after_pos_emb, after_ln_f")
    print(f"  Total hooks: {len(hooks)}")

    return hooks, activation_dict


def compute_gravitational_force(positions):
    """
    Compute gravitational force and related quantities from positions.

    Args:
        positions: array of shape (batch, time, 2) with (x, y) positions

    Returns:
        gravitational_force: dictionary with force components and related quantities
    """
    positions = np.asarray(positions)
    r = np.sqrt(positions[:,:,0]**2 + positions[:,:,1]**2)
    r3 = r**3
    # Avoid division by zero
    r3 = np.where(r3 < 1e-10, 1e-10, r3)
    Fx = -positions[:,:,0] / r3
    Fy = -positions[:,:,1] / r3
    F_magnitude = np.sqrt(Fx**2 + Fy**2)

    # Store gravitational force information
    gravitational_force = {
        'Fx': Fx.flatten(),
        'Fy': Fy.flatten(),
        'F_magnitude': F_magnitude.flatten(),
        'F_direction_x': (Fx / F_magnitude).flatten(),
        'F_direction_y': (Fy / F_magnitude).flatten(),
        # Distance-related quantities
        'r': r.flatten(),                    # r = sqrt(x^2 + y^2)
        'inv_r': (1.0 / np.where(r < 1e-10, 1e-10, r)).flatten(),  # 1/r
        'r_squared': (r**2).flatten(),        # r^2
        'inv_r_squared': (1.0 / np.where(r**2 < 1e-10, 1e-10, r**2)).flatten(),  # 1/r^2
        'inv_r_cubed': (1.0 / np.where(r**3 < 1e-10, 1e-10, r**3)).flatten(),  # 1/r^3
        # Position coordinates
        'x': positions[:,:,0].flatten(),      # x coordinate
        'y': positions[:,:,1].flatten(),      # y coordinate
    }

    return gravitational_force


def collect_activations(model, inputs, targets, activation_dict):
    """
    Run forward pass to collect activations.

    Returns:
        predictions: model predictions
        gravitational_force: computed gravitational force quantities
    """
    model.eval()
    with torch.no_grad():
        predictions, _ = model.forward(inputs, None)

        # Compute gravitational force for comparison
        # Move to CPU if on GPU before converting to numpy
        positions = inputs.cpu().numpy()  # (batch, time, 2)
        gravitational_force = compute_gravitational_force(positions)

    print("Activations collected:")
    for name in activation_dict.keys():
        print(f"  {name}: {activation_dict[name].shape}")

    return predictions, gravitational_force


def run_linear_probes(activation_dict, gravitational_force):
    """
    Run linear probes to test if intermediate representations contain
    linear directions corresponding to gravitational force.

    Returns:
        probe_results: dictionary mapping layer names to probe results
    """
    probe_results = {}

    for layer_name, activations in activation_dict.items():
        # Flatten activations: (batch, time, features) -> (batch*time, features)
        # Move to CPU if on GPU before converting to numpy
        act_flat = activations.reshape(-1, activations.shape[-1]).cpu().numpy()

        # Extract last state activations: (batch, time, features) -> (batch, features)
        batch_size, time_steps, _ = activations.shape
        act_last = activations[:, -1, :].cpu().numpy()  # (batch, features)

        # Probe for different gravitational force components
        probes = {}

        # Define probe targets
        probe_targets = [
            'F_magnitude', 'F_direction_x', 'F_direction_y', 'Fx', 'Fy',
            'r', 'inv_r', 'r_squared', 'inv_r_squared', 'inv_r_cubed', 'x', 'y'
        ]

        for probe_name in probe_targets:
            # Reshape gravitational force back to (batch, time) to extract last timestep
            force_reshaped = gravitational_force[probe_name].reshape(batch_size, time_steps)
            force_last = force_reshaped[:, -1]  # (batch,)
            force_flat = gravitational_force[probe_name]  # Already flattened

            # Probe on all positions
            probe = LinearRegression()
            probe.fit(act_flat, force_flat)
            pred = probe.predict(act_flat)
            r2_all = r2_score(force_flat, pred)

            # Probe on last states only
            probe_last = LinearRegression()
            probe_last.fit(act_last, force_last)
            pred_last = probe_last.predict(act_last)
            r2_last = r2_score(force_last, pred_last)

            probes[probe_name] = {'r2': r2_all, 'r2_last': r2_last}

        probe_results[layer_name] = probes

    # Print results - DISABLED for cleaner training output
    # All probe results are still computed and saved for later analysis
    # print("\nLinear Probe Results (R² scores):")
    # print("=" * 80)
    # for layer_name, probes in probe_results.items():
    #     print(f"\n{layer_name}:")
    #     for probe_name, result in probes.items():
    #         print(f"  {probe_name:20s}: R² (all) = {result['r2']:.4f}, R² (last) = {result['r2_last']:.4f}")

    return probe_results


def run_geometry_probes(activation_dict, orbital_params, trajectory_indices):
    """
    Run linear probes to test if intermediate representations contain
    linear directions corresponding to orbital geometry parameters.

    Args:
        activation_dict: dictionary of activations from the model
        orbital_params: list of dicts with keys: e, a, b, c, average_radius
        trajectory_indices: array of trajectory indices corresponding to each activation

    Returns:
        probe_results: dictionary mapping layer names to probe results
    """
    probe_results = {}

    # Extract orbital parameters for the trajectories we're probing
    # orbital_params is a list, trajectory_indices maps sequence indices to trajectory indices
    # When trajectories are chopped, multiple sequences map to the same trajectory
    # We need to map each activation to its corresponding orbital parameter

    # Create a mapping from trajectory index to orbital parameter
    traj_to_params = {}
    for idx, params in enumerate(orbital_params):
        traj_to_params[idx] = params

    # For each activation, we need to know which trajectory it came from
    # The trajectory_indices array tells us this

    # Extract orbital parameter values for each trajectory
    e_values = np.array([traj_to_params.get(i, {}).get('e', 0.0) for i in trajectory_indices])
    a_values = np.array([traj_to_params.get(i, {}).get('a', 0.0) for i in trajectory_indices])
    b_values = np.array([traj_to_params.get(i, {}).get('b', 0.0) for i in trajectory_indices])
    c_values = np.array([traj_to_params.get(i, {}).get('c', 0.0) for i in trajectory_indices])
    avg_radius_values = np.array([traj_to_params.get(i, {}).get('average_radius', 0.0) for i in trajectory_indices])
    LRL_x_values = np.array([traj_to_params.get(i, {}).get('LRL_x', 0.0) for i in trajectory_indices])
    LRL_y_values = np.array([traj_to_params.get(i, {}).get('LRL_y', 0.0) for i in trajectory_indices])
    LRL_magnitude_values = np.array([traj_to_params.get(i, {}).get('LRL_magnitude', 0.0) for i in trajectory_indices])
    LRL_angle_values = np.array([traj_to_params.get(i, {}).get('LRL_angle', 0.0) for i in trajectory_indices])
    n_x_values = np.array([traj_to_params.get(i, {}).get('n_x', 0.0) for i in trajectory_indices])
    n_y_values = np.array([traj_to_params.get(i, {}).get('n_y', 0.0) for i in trajectory_indices])

    for layer_name, activations in activation_dict.items():
        # Flatten activations: (batch, time, features) -> (batch*time, features)
        # Move to CPU if on GPU before converting to numpy
        act_flat = activations.reshape(-1, activations.shape[-1]).cpu().numpy()

        # Extract last state activations: (batch, time, features) -> (batch, features)
        batch_size, time_steps, _ = activations.shape
        act_last = activations[:, -1, :].cpu().numpy()  # (batch, features)

        # For orbital parameters, we need to repeat the trajectory-level values
        # for each time step in the activation
        # Repeat each trajectory's orbital parameter for all time steps
        e_flat = np.repeat(e_values, time_steps)
        a_flat = np.repeat(a_values, time_steps)
        b_flat = np.repeat(b_values, time_steps)
        c_flat = np.repeat(c_values, time_steps)
        avg_radius_flat = np.repeat(avg_radius_values, time_steps)
        LRL_x_flat = np.repeat(LRL_x_values, time_steps)
        LRL_y_flat = np.repeat(LRL_y_values, time_steps)
        LRL_magnitude_flat = np.repeat(LRL_magnitude_values, time_steps)
        LRL_angle_flat = np.repeat(LRL_angle_values, time_steps)
        n_x_flat = np.repeat(n_x_values, time_steps)
        n_y_flat = np.repeat(n_y_values, time_steps)
        # Compute inverse values
        inv_a_flat = np.repeat(1.0 / a_values, time_steps)
        inv_a2_flat = np.repeat(1.0 / (a_values ** 2), time_steps)
        inv_b_flat = np.repeat(1.0 / b_values, time_steps)
        inv_b2_flat = np.repeat(1.0 / (b_values ** 2), time_steps)

        # For last states, orbital parameters are just the trajectory values (not repeated)
        e_last = e_values
        a_last = a_values
        b_last = b_values
        c_last = c_values
        avg_radius_last = avg_radius_values
        LRL_x_last = LRL_x_values
        LRL_y_last = LRL_y_values
        LRL_magnitude_last = LRL_magnitude_values
        LRL_angle_last = LRL_angle_values
        n_x_last = n_x_values
        n_y_last = n_y_values
        inv_a_last = 1.0 / a_values
        inv_a2_last = 1.0 / (a_values ** 2)
        inv_b_last = 1.0 / b_values
        inv_b2_last = 1.0 / (b_values ** 2)

        # Probe for different orbital geometry parameters
        probes = {}

        # Define probe targets
        probe_targets = ['e', 'a', 'b', 'c', 'average_radius', 'LRL_x', 'LRL_y', 'LRL_magnitude', 'LRL_angle', 'n_x', 'n_y',
                         '1/a', '1/a^2', '1/b', '1/b^2']
        probe_values_flat = {
            'e': e_flat,
            'a': a_flat,
            'b': b_flat,
            'c': c_flat,
            'average_radius': avg_radius_flat,
            'LRL_x': LRL_x_flat,
            'LRL_y': LRL_y_flat,
            'LRL_magnitude': LRL_magnitude_flat,
            'LRL_angle': LRL_angle_flat,
            'n_x': n_x_flat,
            'n_y': n_y_flat,
            '1/a': inv_a_flat,
            '1/a^2': inv_a2_flat,
            '1/b': inv_b_flat,
            '1/b^2': inv_b2_flat
        }
        probe_values_last = {
            'e': e_last,
            'a': a_last,
            'b': b_last,
            'c': c_last,
            'average_radius': avg_radius_last,
            'LRL_x': LRL_x_last,
            'LRL_y': LRL_y_last,
            'LRL_magnitude': LRL_magnitude_last,
            'LRL_angle': LRL_angle_last,
            'n_x': n_x_last,
            'n_y': n_y_last,
            '1/a': inv_a_last,
            '1/a^2': inv_a2_last,
            '1/b': inv_b_last,
            '1/b^2': inv_b2_last
        }

        for probe_name in probe_targets:
            # Probe on all positions
            probe = LinearRegression()
            probe.fit(act_flat, probe_values_flat[probe_name])
            pred = probe.predict(act_flat)
            r2_all = r2_score(probe_values_flat[probe_name], pred)

            # Probe on last states only
            probe_last = LinearRegression()
            probe_last.fit(act_last, probe_values_last[probe_name])
            pred_last = probe_last.predict(act_last)
            r2_last = r2_score(probe_values_last[probe_name], pred_last)

            probes[probe_name] = {'r2': r2_all, 'r2_last': r2_last}

        probe_results[layer_name] = probes

    # Print results
    # Print results - DISABLED for cleaner training output
    # All probe results are still computed and saved for later analysis
    # print("\nGeometry Probe Results (R² scores):")
    # print("=" * 80)
    # for layer_name, probes in probe_results.items():
    #     print(f"\n{layer_name}:")
    #     for probe_name, result in probes.items():
    #         print(f"  {probe_name:20s}: R² (all) = {result['r2']:.4f}, R² (last) = {result['r2_last']:.4f}")

    return probe_results
