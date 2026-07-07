#!/usr/bin/env python
"""
mechanistic_analysis.py

Decompose the trained model's next-step prediction into per-input-position
contributions, using the embedding-level ELM Jacobian validated in
verify_elm.py.

For each sampled real-orbit window x* of length T, the embedding-level
decomposition gives the EXACT identity (recon_err ~ 1e-15):

    y* = Sum_t J_{h,t} @ h_t
       = Sum_t (J_{h,t} W_in) @ x*_t  +  Sum_t J_{h,t} @ wpe[t]
         \-------- content --------/     \------ positional ------/

We restrict the Jacobian rows to the LAST output position -- the model's
next-step prediction given the full window -- and read off, per input
position t:

    content_t    = J^last_{x,t} @ x*_t        (how much the (x,y) at time t
                                               drives the prediction)
    positional_t = J^last_{h,t} @ wpe[t]      (how much position t's encoding
                                               drives the prediction)

These sum exactly to the prediction: Sum_t (content_t + positional_t) = y_pred.
The magnitudes ||content_t|| vs t are the mechanistic signal:

    block_size=2  (Newtonian) : contribution concentrated on recent positions
    block_size=100 (Keplerian): contribution spread across the trajectory

Prints stats to stdout; saves one contribution-vs-position plot per checkpoint
to figures/mechanistic_contributions_block_size_{bs}.png.

Usage:
    python mechanistic_analysis.py
"""

import io
import os
import math
import contextlib

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt   # backend left untouched on import so notebooks keep
                                  # their inline backend; main() selects Agg for CLI runs

from . import model as M
from .model import load_from_checkpoint
from .data import (
    load_trajectories,
    chop_trajectories_into_sequences,
)

# float64 -- the embedding-level reconstruction is exact only at full precision.
torch.set_default_dtype(torch.float64)

# Keep the repo's (fixed, mean-live) LayerNorm detachment.
FIXED_LN = M.layernorm_detached


def load_quiet(checkpoint_path):
    """Load a checkpoint, suppressing the model's construction prints."""
    M.layernorm_detached = FIXED_LN
    with contextlib.redirect_stdout(io.StringIO()):
        model, ckpt = load_from_checkpoint(checkpoint_path, device='cpu')
    model.eval()
    return model, ckpt


def sample_orbit_windows(block_size, n_windows, seed=0, data_dir='data_cv', offset=0):
    """
    Sample distinct real-orbit windows of length block_size.

    Identical sampling path to verify_elm.real_orbit_scatter: load real
    trajectories (num_traj, 100, 2), chop into length-block_size windows, then
    draw `n_windows` distinct windows with a seeded RNG.

    offset: skip this many trajectories before sampling. Training splits the
        dataset 50/50 by trajectory (train = first half, test = second half), so
        pass offset=0 for train orbits (default) and offset=train_size (e.g.
        10000) to analyse held-out test orbits instead.
    """
    seqs_per_traj = max(1, 100 // block_size)
    n_traj_need = max(50, math.ceil(n_windows / seqs_per_traj) + 5)
    with contextlib.redirect_stdout(io.StringIO()):
        trajs = load_trajectories(data_dir, num_trajectories_needed=offset + n_traj_need)
        trajs = trajs[offset:]   # keep only the requested split's trajectories
        assert trajs.shape[0] > 0, f"offset={offset} exceeds available trajectories"
        # chop needs block_size + 1 points to also carve a next-step target; when
        # block_size == trajectory length that leaves zero windows. We only need
        # the input windows here, so cap the length to leave room for the +1.
        eff_block_size = min(block_size, trajs.shape[1] - 1)
        inputs, _ = chop_trajectories_into_sequences(trajs, eff_block_size, seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.choice(inputs.shape[0],
                     size=min(n_windows, inputs.shape[0]), replace=False)
    return inputs[idx]   # (n_windows, block_size, 2)


def window_contributions(model, x_star):
    """
    Per-input-position contribution to the LAST-position prediction.

    Returns:
        content_vec: (T, D) tensor -- content_t = J^last_{x,t} @ x*_t
        pos_vec:     (T, D) tensor -- positional_t = J^last_{h,t} @ wpe[t]
        pred:        (D,)   tensor -- model's last-position prediction (y_star[last])
        recon_err:   float        -- validity of J_h (should be ~1e-15)
    """
    (jac_h, jac_x, y_star, recon_err, _, _, _, _) = \
        model.compute_detached_jacobians_embedding(x_star)

    T = x_star.shape[1]
    D = x_star.shape[2]
    last = slice((T - 1) * D, T * D)        # rows of flattened output -> last position

    pos = torch.arange(T)
    wpe = model.transformer.wpe(pos)        # (T, n_embd)

    content_vec = torch.stack([
        (jac_x[t][last] @ x_star[0, t]).detach() for t in range(T)
    ])                                      # (T, D)
    pos_vec = torch.stack([
        (jac_h[t][last] @ wpe[t]).detach() for t in range(T)
    ])                                      # (T, D)
    pred = y_star.reshape(T, D)[T - 1].detach()   # (D,)
    return content_vec, pos_vec, pred, recon_err


def analyze_checkpoint(checkpoint_path, block_size, n_windows=200, seed=0):
    """Aggregate per-position contributions across sampled windows."""
    model, ckpt = load_quiet(checkpoint_path)
    windows = sample_orbit_windows(block_size, n_windows, seed=seed)

    content_mags = []   # (n, T) ||content_t||
    pos_mags = []       # (n, T) ||positional_t||
    recon_errs = []     # validity of J_h per window
    sum_errs = []       # ||(sum content + sum pos) - pred|| -- additive-decomp check

    for w in windows:
        x_star = torch.tensor(w, dtype=torch.float64).unsqueeze(0)   # (1, T, 2)
        cvec, pvec, pred, re = window_contributions(model, x_star)
        content_mags.append(cvec.norm(dim=1).numpy())   # (T,)
        pos_mags.append(pvec.norm(dim=1).numpy())
        recon_errs.append(re)
        sum_errs.append((cvec.sum(0) + pvec.sum(0) - pred).norm().item())

    return {
        'ckpt': ckpt,
        'n_windows': len(windows),
        'content_mags': np.array(content_mags),   # (n, T)
        'pos_mags': np.array(pos_mags),            # (n, T)
        'recon_err_max': float(np.max(recon_errs)),
        'sum_err_max': float(np.max(sum_errs)),
    }


def report_and_plot(res, block_size):
    cm = res['content_mags']          # (n, T)
    pm = res['pos_mags']
    T = cm.shape[1]
    positions = np.arange(T)

    cm_mean, cm_std = cm.mean(0), cm.std(0)
    pm_mean = pm.mean(0)
    # Per-window normalized content profile (scale-invariant shape).
    frac = cm / (cm.sum(1, keepdims=True) + 1e-30)
    frac_mean, frac_std = frac.mean(0), frac.std(0)

    print(f"\n  Validity (must hold for the contributions to be meaningful):")
    print(f"    max J_h recon_err            = {res['recon_err_max']:.2e}  (embedding-level Jacobian exact)")
    print(f"    max ||Sum_t(content+pos) - pred|| = {res['sum_err_max']:.2e}  (decomposition sums to the prediction)")

    print(f"\n  Per-input-position contribution to the next-step prediction")
    print(f"  (mean over {res['n_windows']} real-orbit windows):")
    print(f"    {'pos t':>6}  {'||content||':>14}  {'||positional||':>15}  {'content frac':>13}")
    for t in range(T):
        print(f"    {t:>6}  {cm_mean[t]:>14.4e}  {pm_mean[t]:>15.4e}  {frac_mean[t]:>13.3f}")

    # most-recent vs oldest content contribution
    print(f"\n    most-recent (t={T-1}) content fraction = {frac_mean[T-1]:.3f}")
    print(f"    oldest      (t=0)   content fraction = {frac_mean[0]:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(positions, cm_mean, '-o', color='tab:blue',
             label=r'content  $\|J^{last}_{x,t}\,x^*_t\|$')
    ax1.fill_between(positions, cm_mean - cm_std, cm_mean + cm_std,
                     color='tab:blue', alpha=0.2)
    ax1.plot(positions, pm_mean, '-s', color='tab:orange',
             label=r'positional  $\|J^{last}_{h,t}\,wpe[t]\|$')
    ax1.set_xlabel('input position t  (0 = oldest, T-1 = most recent)')
    ax1.set_ylabel('contribution magnitude to prediction')
    ax1.set_title(f'Per-position contribution (block_size={block_size})')
    ax1.legend(fontsize=8)

    ax2.plot(positions, frac_mean, '-o', color='tab:blue')
    ax2.fill_between(positions, frac_mean - frac_std, frac_mean + frac_std,
                     color='tab:blue', alpha=0.2)
    ax2.set_xlabel('input position t  (0 = oldest, T-1 = most recent)')
    ax2.set_ylabel('fraction of total content contribution')
    ax2.set_title('Normalized content profile')

    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out_path = f'figures/mechanistic_contributions_block_size_{block_size}.png'
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\n  saved plot -> {out_path}")


def main():
    matplotlib.use('Agg')   # headless: write PNGs, never open a window
    block_sizes = [2, 100]
    num_traj = 10000
    noise_scale = 0.1
    checkpoint_dir = './checkpoints/kepler_cv_blocksize_elm'
    n_windows = 200

    print("=" * 78)
    print("MECHANISTIC ANALYSIS: input-space contributions via embedding-level ELM")
    print("=" * 78)

    for block_size in block_sizes:
        checkpoint_path = (
            f'{checkpoint_dir}/model_block_size_{block_size}'
            f'_num_trajectories_{num_traj}_noise_scale_{noise_scale}.pt'
        )

        print(f"\n{'=' * 78}")
        print(f"Block size = {block_size}")
        print(f"{'=' * 78}")

        if not os.path.exists(checkpoint_path):
            print(f"  (no checkpoint at {checkpoint_path} -- skipping)")
            continue

        res = analyze_checkpoint(checkpoint_path, block_size,
                                 n_windows=n_windows, seed=0)
        step = res['ckpt'].get('step', 'unknown')
        print(f"  Checkpoint: step {step}   ({res['n_windows']} windows sampled)")
        report_and_plot(res, block_size)

    print(f"\n{'=' * 78}")
    print("Each prediction is split additively across input positions through the")
    print("same x*-frozen Jacobian. Compare the normalized profile across block")
    print("sizes: concentrated on recent positions (Newtonian) vs spread across")
    print("the trajectory (Keplerian).")
    print("=" * 78)


if __name__ == "__main__":
    main()
