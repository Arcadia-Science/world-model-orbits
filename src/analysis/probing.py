"""Linear-probing suite (Liu et al. Fig 6 reproduction).

Registers activation hooks, collects hidden states, and regresses Newtonian
(force) and Keplerian (orbital-geometry) target variables against them to
measure what each model linearly encodes.
"""

# ---------------------------------------------------------------------------
# Adapted from Liu et al., "From Kepler to Newton: Inductive Biases Guide
# Learned World Models in Transformers" (arXiv:2602.06923), CC BY 4.0
# (https://creativecommons.org/licenses/by/4.0/).
# Source: https://github.com/KindXiaoming/newton-kepler
# Modification: linear/geometry probing extracted from the upstream training
# script. Full attribution in NOTICE.md.
# ---------------------------------------------------------------------------

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .data import sample_windows


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


def _fit_r2(activations, targets):
    """In-sample R^2 of a least-squares linear readout of `targets` from `activations`."""
    probe = LinearRegression().fit(activations, targets)
    return r2_score(targets, probe.predict(activations))


def _probe_layers(activation_dict, make_targets):
    """Fit linear probes for a set of targets at every hooked site.

    The probing protocol is the same whatever is being decoded -- for each layer,
    read the activations two ways (every position, and the last position only) and
    score a least-squares readout of each target against both. Only how the targets
    are assembled differs, which is what `make_targets` supplies.

    Scoring is in-sample, matching the protocol this replicates; see
    `run_linear_probes` for why.

    Args:
        activation_dict: layer name -> (batch, time, features) activations
        make_targets:    (batch_size, time_steps) -> {name: (values_flat, values_last)},
                         where values_flat is per (batch*time) and values_last per batch

    Returns:
        {layer_name: {target_name: {'r2': ..., 'r2_last': ...}}}
    """
    probe_results = {}
    for layer_name, activations in activation_dict.items():
        # (batch, time, features) -> every position, then the last position only
        act_flat = activations.reshape(-1, activations.shape[-1]).cpu().numpy()
        batch_size, time_steps, _ = activations.shape
        act_last = activations[:, -1, :].cpu().numpy()

        probe_results[layer_name] = {
            name: {'r2': _fit_r2(act_flat, y_flat), 'r2_last': _fit_r2(act_last, y_last)}
            for name, (y_flat, y_last) in make_targets(batch_size, time_steps).items()
        }
    return probe_results


# Newtonian targets: the force itself, its direction, and the radial quantities it
# is built from -- the last of which act as a sanity floor on the force result.
FORCE_TARGETS = ['F_magnitude', 'F_direction_x', 'F_direction_y', 'Fx', 'Fy',
                 'r', 'inv_r', 'r_squared', 'inv_r_squared', 'inv_r_cubed', 'x', 'y']

# Keplerian targets: whole-orbit geometry, constant along a trajectory.
GEOMETRY_TARGETS = ['e', 'a', 'b', 'c', 'average_radius', 'LRL_x', 'LRL_y',
                    'LRL_magnitude', 'LRL_angle', 'n_x', 'n_y']


def run_linear_probes(activation_dict, gravitational_force):
    """Probe every hooked site for the Newtonian force variables.

    Scoring is in-sample, and deliberately so: this replicates the protocol of
    Liu et al., which fits and scores on the same activations with no held-out
    split. The targets vary per position, so the last-position values are read
    straight out of the flattened array.
    """
    def make_targets(batch_size, time_steps):
        targets = {}
        for name in FORCE_TARGETS:
            values_flat = gravitational_force[name]
            values_last = values_flat.reshape(batch_size, time_steps)[:, -1]
            targets[name] = (values_flat, values_last)
        return targets

    return _probe_layers(activation_dict, make_targets)


def run_geometry_probes(activation_dict, orbital_params, trajectory_indices):
    """Probe every hooked site for the Keplerian orbital-geometry parameters.

    Unlike the force variables, these are properties of the whole orbit and so are
    constant along a trajectory: each window's value is its source trajectory's, and
    the flattened form is that value repeated across the window's positions.

    Args:
        activation_dict:    layer name -> (batch, time, features) activations
        orbital_params:     list of dicts, one per trajectory (keys: e, a, b, c, ...)
        trajectory_indices: source trajectory of each activation row. Indices outside
                            the orbital_params table score as 0.0 rather than raising,
                            since callers may probe more windows than there are entries.
    """
    n_params = len(orbital_params)

    def gather(key):
        return np.array([
            orbital_params[i].get(key, 0.0) if 0 <= i < n_params else 0.0
            for i in trajectory_indices
        ])

    values = {key: gather(key) for key in GEOMETRY_TARGETS}
    # Reciprocal axis lengths: a Keplerian model that encodes the orbit scale may
    # do so through 1/a rather than a (the mean motion goes as a^-3/2).
    values['1/a']   = 1.0 / values['a']
    values['1/a^2'] = 1.0 / values['a'] ** 2
    values['1/b']   = 1.0 / values['b']
    values['1/b^2'] = 1.0 / values['b'] ** 2

    def make_targets(batch_size, time_steps):
        return {name: (np.repeat(v, time_steps), v) for name, v in values.items()}

    return _probe_layers(activation_dict, make_targets)


def make_probe_windows(trajectories, orbital_params, block_size, n_probe, seed=0):
    """`n_probe` windows in float64, with each window's source-trajectory index.

    Thin wrapper over `data.sample_windows`: the probes need float64 activations
    and a trajectory index clipped to the orbital-parameter table, which may be
    shorter than the trajectory array.
    """
    windows, traj_idx = sample_windows(trajectories, block_size, n_probe, seed=seed)
    return windows.astype(np.float64), np.clip(traj_idx, 0, len(orbital_params) - 1)


def probe_model(model, trajectories, orbital_params, block_size, n_probe=1000, seed=0):
    """Run Newtonian (force) + Keplerian (geometry) linear probes on one model.

    Returns (newton_results, kepler_results), each a dict keyed by layer name.
    """
    windows, traj_idx = make_probe_windows(trajectories, orbital_params, block_size, n_probe, seed)
    hooks, act = setup_activation_hooks(model)
    _, gforce = collect_activations(model, torch.tensor(windows), None, act)
    newton = run_linear_probes(act, gforce)
    kepler = run_geometry_probes(act, orbital_params, traj_idx)
    for h in hooks:
        h.remove()
    return newton, kepler


def best_r2(probe_results, target, key='r2_last'):
    """Best-over-layers R^2 for a probe target (matches Liu et al. Fig 6 'postfix=_last')."""
    return max(layer[target][key] for layer in probe_results.values() if target in layer)
