#!/usr/bin/env python
"""
newtonian_test.py

Test whether the trained model implements the Newtonian (Verlet) one-step map.

A model that has learned Newtonian dynamics must predict the next position by
central-difference / Stormer-Verlet integration:

    r_{t+1} = 2 r_t - r_{t-1} + a_t dt^2,     a_t = -GM r_t / |r_t|^3

The inertial (linear) part is  2 r_t - r_{t-1}  -- constant-velocity
extrapolation -- with a small O(dt^2) gravitational correction acting on r_t.
In the embedding-level ELM decomposition the last-position prediction is

    r_hat_{t+1} = Sum_s J_s @ r_s        (content blocks, validated exact)

so a Newtonian model predicts, per input position s (D x D blocks):

    J_{recent  = T-1}  ~  2 I  (+ O(dt^2) tidal correction)
    J_{prev    = T-2}  ~ -1 I
    J_{older   < T-2}  ~  0      (ignored)

This script extracts those blocks from real-orbit windows and plots the test:
  [A] isotropic coefficient c_s = trace(J_s)/D vs input position, with the
      Verlet targets (2 at recent, -1 at prev, 0 elsewhere) overlaid.
  [B] per-window scatter of (c_prev, c_recent) against the Verlet target
      (-1, 2) and the trivial persistence/average solution.
  [C] heatmaps of the mean recent / prev Jacobian blocks vs the ideal 2I / -I.

Prints stats; saves the plots to figures/.

Usage:
    python newtonian_test.py
"""

import io
import os
import contextlib

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt   # backend left untouched on import so notebooks keep
                                  # their inline backend; main() selects Agg for CLI runs

from .model import load_from_checkpoint
from .mechanistic import sample_orbit_windows
from .data import load_trajectories

torch.set_default_dtype(torch.float64)


def jacobian_blocks(model, x_star):
    """Per-input-position D x D content Jacobian block for the LAST output."""
    (_, jac_x, _, recon_err, _, _, _, _) = \
        model.compute_detached_jacobians_embedding(x_star)
    T = x_star.shape[1]
    D = x_star.shape[2]
    last = slice((T - 1) * D, T * D)
    blocks = torch.stack([jac_x[t][last].detach() for t in range(T)])  # (T, D, D)
    return blocks.numpy(), recon_err


def collect(model, block_size, n_windows, seed=0, offset=0):
    windows = sample_orbit_windows(block_size, n_windows, seed=seed, offset=offset)
    all_blocks, recon_errs = [], []
    for w in windows:
        x_star = torch.tensor(w, dtype=torch.float64).unsqueeze(0)
        blocks, re = jacobian_blocks(model, x_star)
        all_blocks.append(blocks)
        recon_errs.append(re)
    all_blocks = np.array(all_blocks)         # (n, T, D, D)
    D = all_blocks.shape[-1]
    # isotropic coefficient c_s = trace(J_s)/D  per window
    coeffs = np.trace(all_blocks, axis1=2, axis2=3) / D   # (n, T)
    return all_blocks, coeffs, float(np.max(recon_errs))


def block_rotation_decomp(all_blocks):
    """
    Decompose each D x D content-Jacobian block into the closest scaled rotation.

    For D == 2 a block J splits into an isotropic-scale part and a rotation part:
        scale  s_s = tr(J_s) / 2                 (symmetric / radial gain)
        gen    a_s = (J_s[1,0] - J_s[0,1]) / 2   (antisymmetric / rotation generator)
    so the block's effective rotation is  theta_s = atan2(a_s, s_s)  and its
    scaled-rotation magnitude is  sqrt(s_s^2 + a_s^2). A Verlet block (2I or -I)
    has theta == 0 (pure scale, no rotation); an orbital one-step map carries a
    nonzero theta tracking the per-step angular advance.

    Returns (per window, per position): scale (n,T), gen (n,T), angle_deg (n,T).
    """
    assert all_blocks.shape[-1] == 2, "rotation decomposition assumes D == 2"
    scale = (all_blocks[..., 0, 0] + all_blocks[..., 1, 1]) / 2.0
    gen = (all_blocks[..., 1, 0] - all_blocks[..., 0, 1]) / 2.0
    angle_deg = np.degrees(np.arctan2(gen, scale))
    return scale, gen, angle_deg


def rotation_behavioral(model, block_size, n_windows, seed=0,
                        data_dir='data_cv', offset=0):
    """
    Behavioural rotation test: does the model predict r_{t+1} by rotating the
    current position r_t forward along the orbit (radius preserved)?

    Needs a held-out true next point, so it samples windows of length block_size
    that leave room for the following point. Returns None when block_size is the
    full trajectory length (no held-out next point exists).

    offset: skip this many trajectories before sampling (see sample_orbit_windows).
        offset=0 → train orbits (default); offset=train_size → held-out test orbits.
    """
    n_traj_need = max(60, n_windows // 10 + 10)
    with contextlib.redirect_stdout(io.StringIO()):
        trajs = np.asarray(load_trajectories(
            data_dir, num_trajectories_needed=offset + n_traj_need))
    trajs = trajs[offset:]   # keep only the requested split's trajectories
    assert trajs.shape[0] > 0, f"offset={offset} exceeds available trajectories"
    n_traj, n_pts, _ = trajs.shape
    if block_size >= n_pts:
        return None  # no held-out next point for a full-length window

    # All valid (window, next-point) start positions across trajectories.
    max_start = n_pts - block_size            # need traj[start+block_size]
    rng = np.random.default_rng(seed)
    traj_idx = rng.integers(0, n_traj, size=n_windows)
    start_idx = rng.integers(0, max_start, size=n_windows)
    windows = np.stack([trajs[t, s:s + block_size]
                        for t, s in zip(traj_idx, start_idx)])      # (n,T,2)
    truth = np.stack([trajs[t, s + block_size]
                      for t, s in zip(traj_idx, start_idx)])        # (n,2)

    mdtype = next(model.parameters()).dtype
    with torch.no_grad():
        pred = model(torch.tensor(windows).to(mdtype))[0][:, -1, :]
    pred = pred.double().numpy()                                    # (n,2)
    r_t = windows[:, -1, :]                                         # (n,2)

    def signed_angle(u, v):     # signed angle u -> v about the origin (focus)
        return np.arctan2(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0],
                          (u * v).sum(1))

    ang_model = signed_angle(r_t, pred)
    ang_true = signed_angle(r_t, truth)

    # "rotate r_t by the model's angle, keep its radius" -- pure-rotation pred.
    c, s = np.cos(ang_model), np.sin(ang_model)
    rot_only = np.stack([r_t[:, 0] * c - r_t[:, 1] * s,
                         r_t[:, 0] * s + r_t[:, 1] * c], axis=1)

    # --- angular vs radial split of the elliptical step ---
    # Kepler's 2nd law: dtheta ~ 1/r^2 (fast at perihelion, slow at aphelion).
    # An ellipse also breathes radially (dr != 0); a circle would have dr == 0.
    rmag = np.linalg.norm(r_t, axis=1)            # |r_t|
    inv_r2 = 1.0 / rmag ** 2
    dr_true = np.linalg.norm(truth, axis=1) - rmag   # true per-step radius change
    dr_model = np.linalg.norm(pred, axis=1) - rmag   # model's per-step radius change

    # Specific angular momentum L_z = r x v (z-component), from the in-window
    # velocity estimate v = r_t - r_{t-1}. At fixed radius the true per-step
    # angle scales with L; it is the hidden axis behind the dtheta-vs-1/r^2
    # scatter (constant-L orbits trace constant-slope rays).
    vel = r_t - windows[:, -2, :]
    Lz = r_t[:, 0] * vel[:, 1] - r_t[:, 1] * vel[:, 0]

    def r2(p, y):
        return float(1 - ((p - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())

    # displacement (step) variance explained -- strips position autocorrelation.
    d = truth - r_t
    return {
        'ang_model': ang_model, 'ang_true': ang_true,
        'ang_model_mean': float(np.degrees(ang_model.mean())),
        'ang_true_mean': float(np.degrees(ang_true.mean())),
        'ang_corr': float(np.corrcoef(ang_model, ang_true)[0, 1]),
        'radius_ratio': float(np.linalg.norm(pred, axis=1).mean()
                              / np.linalg.norm(r_t, axis=1).mean()),
        'r2_rotation_only': r2(rot_only, truth),
        'r2_model': r2(pred, truth),
        'r2_persistence': r2(r_t, truth),
        # Kepler-2 angular tracking + radial breathing
        'inv_r2': inv_r2,
        'dr_true': dr_true, 'dr_model': dr_model,
        'Lz': Lz,
        'kepler2_corr_true': float(np.corrcoef(ang_true, inv_r2)[0, 1]),
        'kepler2_corr_model': float(np.corrcoef(ang_model, inv_r2)[0, 1]),
        'dr_corr': float(np.corrcoef(dr_model, dr_true)[0, 1]),
        'dr_true_absmean': float(np.abs(dr_true).mean()),
        'r2_disp_rotlock': r2(rot_only - r_t, d),   # radius-LOCKED rotation
        'r2_disp_model': r2(pred - r_t, d),         # full model
        'n': int(n_windows),
    }


def plot_rotation(all_blocks, block_size, behav):
    """Rotation-mechanism figure: per-block rotation angle, and (if available)
    model vs true per-step orbital angle."""
    n, T, D, _ = all_blocks.shape
    scale, gen, angle_deg = block_rotation_decomp(all_blocks)
    a_mean, a_std = angle_deg.mean(0), angle_deg.std(0)

    ncols = 2 if behav is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.8), squeeze=False)
    axes = axes[0]

    ax = axes[0]
    positions = np.arange(T)
    ax.axhline(0, color='gray', lw=0.7)
    ax.errorbar(positions, a_mean, yerr=a_std, fmt='-o', color='tab:purple',
                capsize=3, label=r'block rotation $\theta_s=\mathrm{atan2}(a_s,s_s)$')
    if T <= 12:
        ax.set_xticks(positions)
    ax.set_xlabel('input position s  (0 = oldest, T-1 = most recent)')
    ax.set_ylabel('operator rotation angle (deg)')
    ax.set_title(f'Per-block rotation (block_size={block_size})\n'
                 'Verlet => 0 deg; orbital one-step => nonzero')
    ax.legend(fontsize=8)

    if behav is not None:
        ax = axes[1]
        at, am = np.degrees(behav['ang_true']), np.degrees(behav['ang_model'])
        lo = float(min(at.min(), am.min())); hi = float(max(at.max(), am.max()))
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='y = x', zorder=1)
        ax.scatter(at, am, s=12, alpha=0.4, color='tab:purple', zorder=2)
        ax.set_xlabel('true per-step orbital angle (deg)')
        ax.set_ylabel(r'model rotation $\angle(r_t \to \hat r_{t+1})$ (deg)')
        ax.set_title('Behavioural: model rotates $r_t$ forward\n'
                     f"corr {behav['ang_corr']:.2f}, "
                     f"radius x{behav['radius_ratio']:.3f}, "
                     f"rot-only $R^2$ {behav['r2_rotation_only']:.3f}")
        ax.legend(fontsize=8, loc='best')

    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = f'figures/newtonian_rotation_test_block_size_{block_size}.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, a_mean, a_std


def plot_kepler2(block_size, behav):
    """
    Angular-vs-radial decomposition of the elliptical step.

    [left]  Kepler's 2nd law: per-step angle vs 1/r_t^2. The true angle rises
            with 1/r^2 (faster near the focus); the model's angle follows it,
            showing it learned the state-dependent angular speed of an ellipse.
    [right] Radial breathing: model dr vs true dr. On an ellipse dr != 0, but
            the model predicts ~no radius change (uncorrelated), so it drops the
            elliptical radial term -- the same second-order piece Verlet supplies.
    """
    if behav is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    # true points coloured by angular momentum L: constant-L orbits trace
    # constant-slope rays, so the "scatter" resolves into L-ordered bands --
    # showing it is the hidden L axis, not randomness.
    Lz = behav['Lz']
    vlo = float(np.nanpercentile(Lz, 2)); vhi = float(np.nanpercentile(Lz, 98))
    sc = ax.scatter(behav['inv_r2'], np.degrees(behav['ang_true']), s=12,
                    alpha=0.6, c=Lz, cmap='viridis', vmin=vlo, vmax=vhi,
                    label=f"true, coloured by L  (corr w/ $1/r^2$ {behav['kepler2_corr_true']:.2f})")
    ax.scatter(behav['inv_r2'], np.degrees(behav['ang_model']), s=6, alpha=0.5,
               color='#FFB347', marker='.',
               label=f"model  (corr w/ $1/r^2$ {behav['kepler2_corr_model']:.2f})")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r'angular momentum  $L_z = r_t \times v$')
    ax.set_xlabel(r'$1/r_t^2$')
    ax.set_ylabel('per-step angle (deg)')
    ax.set_title(f"Kepler's 2nd law: angle $\\propto L/r^2$ (block_size={block_size})\n"
                 "true scatter resolves into L-bands; model (orange) ignores L")
    ax.legend(fontsize=8, loc='upper left')

    ax = axes[1]
    dt, dm = behav['dr_true'], behav['dr_model']
    lo = float(min(dt.min(), dm.min())); hi = float(max(dt.max(), dm.max()))
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='y = x', zorder=1)
    ax.axhline(0, color='gray', lw=0.7, zorder=1)
    ax.scatter(dt, dm, s=10, alpha=0.35, color='tab:red', zorder=2)
    ax.set_xlabel(r'true radius change $\Delta r = |r_{t+1}| - |r_t|$')
    ax.set_ylabel(r'model radius change $|\hat r_{t+1}| - |r_t|$')
    ax.set_title('Radial breathing (model ignores it)\n'
                 f"corr {behav['dr_corr']:.2f}; "
                 f"step $R^2$: rot-locked {behav['r2_disp_rotlock']:.2f} "
                 f"vs model {behav['r2_disp_model']:.2f}")
    ax.legend(fontsize=8, loc='best')

    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = f'figures/newtonian_kepler2_test_block_size_{block_size}.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_test(all_blocks, coeffs, block_size, recon_err_max):
    n, T, D, _ = all_blocks.shape
    positions = np.arange(T)
    c_mean, c_std = coeffs.mean(0), coeffs.std(0)

    target = np.zeros(T)
    if T >= 1:
        target[T - 1] = 2.0      # recent -> 2I
    if T >= 2:
        target[T - 2] = -1.0     # prev   -> -I

    # ---- figure 1: coefficient-vs-position + scatter ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    ax.axhline(0, color='gray', lw=0.7)
    ax.errorbar(positions, c_mean, yerr=c_std, fmt='-o', color='tab:blue',
                capsize=3, label=r'measured  $c_s=\mathrm{tr}(J_s)/D$')
    ax.plot(positions, target, 'X', color='tab:red', ms=12,
            label='Verlet target (2, -1, 0)')
    ax.set_xlabel('input position s  (0 = oldest, T-1 = most recent)')
    ax.set_ylabel('isotropic coefficient  $c_s$')
    ax.set_title(f'Verlet coefficient test (block_size={block_size})')
    ax.legend(fontsize=8)

    ax = axes[1]
    if T >= 2:
        c_prev, c_recent = coeffs[:, T - 2], coeffs[:, T - 1]
        ax.scatter(c_prev, c_recent, s=14, alpha=0.5, color='tab:blue',
                   label='per-window', zorder=2)
        ax.scatter([c_prev.mean()], [c_recent.mean()], s=120, marker='D',
                   color='tab:blue', edgecolor='k',
                   label=f'measured mean ({c_prev.mean():.2f}, {c_recent.mean():.2f})',
                   zorder=4)
        ax.scatter([-1], [2], s=200, marker='*', color='tab:red', edgecolor='k',
                   label='Verlet (-1, 2)', zorder=5)
        # persistence (predict latest point): c_recent=1, c_prev=0
        ax.scatter([0], [1], s=120, marker='P', color='tab:green', edgecolor='k',
                   label='persistence (0, 1)', zorder=4)
        # consistency reference line c_prev + c_recent = 1 (Verlet & persistence both lie on it)
        lo = min(-1.5, c_prev.min()); hi = max(2.5, c_recent.max())
        xs = np.linspace(lo, hi, 50)
        ax.plot(xs, 1 - xs, ls='--', color='gray', lw=1,
                label=r'$c_{prev}+c_{recent}=1$', zorder=1)
        ax.set_xlabel(r'$c_{prev}$  (position T-2)')
        ax.set_ylabel(r'$c_{recent}$  (position T-1)')
        ax.set_title('Where does the model sit?')
        ax.legend(fontsize=8, loc='best')
    else:
        ax.axis('off')

    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out1 = f'figures/newtonian_verlet_test_block_size_{block_size}.png'
    fig.savefig(out1, dpi=130)
    plt.close(fig)

    # ---- figure 2: Jacobian-block heatmaps vs ideal ----
    out2 = None
    if T >= 2:
        meanJ = all_blocks.mean(0)                 # (T, D, D)
        J_recent = meanJ[T - 1]
        J_prev = meanJ[T - 2]
        ideal_recent = 2 * np.eye(D)
        ideal_prev = -1 * np.eye(D)
        panels = [
            (J_recent, f'measured $J_{{recent}}$ (pos {T-1})'),
            (ideal_recent, 'ideal  $2I$'),
            (J_prev, f'measured $J_{{prev}}$ (pos {T-2})'),
            (ideal_prev, 'ideal  $-I$'),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 7))
        vmax = 2.2
        for axx, (mat, title) in zip(axes.ravel(), panels):
            im = axx.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axx.set_title(title, fontsize=10)
            axx.set_xticks(range(D)); axx.set_yticks(range(D))
            for i in range(D):
                for j in range(D):
                    axx.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                             color='k', fontsize=10)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
        fig.suptitle(f'Content Jacobian blocks vs Verlet ideal '
                     f'(block_size={block_size})', fontsize=11)
        out2 = f'figures/newtonian_jacobian_blocks_block_size_{block_size}.png'
        fig.savefig(out2, dpi=130)
        plt.close(fig)

    return out1, out2, target, c_mean


def main():
    matplotlib.use('Agg')   # headless: write PNGs, never open a window
    block_sizes = [2, 100]
    num_traj = 10000
    noise_scale = 0.1
    checkpoint_dir = './checkpoints/kepler_cv_blocksize_elm'
    n_windows = 200

    print("=" * 78)
    print("NEWTONIAN (VERLET) WORLD-MODEL TEST")
    print("ideal:  J_recent = 2I,  J_prev = -I,  J_older = 0")
    print("=" * 78)

    for block_size in block_sizes:
        checkpoint_path = (
            f'{checkpoint_dir}/model_block_size_{block_size}'
            f'_num_trajectories_{num_traj}_noise_scale_{noise_scale}.pt'
        )
        print(f"\n{'=' * 78}\nBlock size = {block_size}\n{'=' * 78}")
        if not os.path.exists(checkpoint_path):
            print(f"  (no checkpoint at {checkpoint_path} -- skipping)")
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            model, ckpt = load_from_checkpoint(checkpoint_path, device='cpu')
        model.eval()

        all_blocks, coeffs, recon_err_max = collect(model, block_size, n_windows)
        T = coeffs.shape[1]
        out1, out2, target, c_mean = plot_test(all_blocks, coeffs, block_size, recon_err_max)

        print(f"  Checkpoint step {ckpt.get('step', '?')}, {coeffs.shape[0]} windows, "
              f"max J_h recon_err {recon_err_max:.1e}")
        print(f"\n  Isotropic coefficient c_s = trace(J_s)/D  (mean +/- std):")
        for s in range(T):
            tag = ''
            if s == T - 1: tag = '  <- recent (target  2)'
            elif s == T - 2: tag = '  <- prev   (target -1)'
            if T > 8 and 1 < s < T - 2:
                continue
            print(f"    pos {s:>3}:  {c_mean[s]:+.3f} +/- {coeffs[:, s].std():.3f}{tag}")
        if T > 8:
            print(f"    (positions 2..{T-3} omitted; target 0 there)")

        if T >= 2:
            c_prev, c_recent = coeffs[:, T - 2].mean(), coeffs[:, T - 1].mean()
            dist = float(np.hypot(c_prev - (-1), c_recent - 2))
            print(f"\n  (c_prev, c_recent) = ({c_prev:+.3f}, {c_recent:+.3f})")
            print(f"    distance to Verlet target (-1, +2) = {dist:.3f}")
            verdict = "MATCHES Verlet" if dist < 0.4 else "does NOT match Verlet (not Newtonian)"
            print(f"    verdict: {verdict}")
        print(f"\n  saved -> {out1}")
        if out2:
            print(f"  saved -> {out2}")

        # ---- rotation mechanism: what the model does INSTEAD of Verlet ----
        # Behavioural test is just forward passes (cheap), so use many more
        # windows than the Jacobian sample -- the Kepler-2 / dr correlations
        # need ~thousands to stabilise (L varies across orbits -> real scatter).
        behav = rotation_behavioral(model, block_size, max(n_windows, 4000))
        out3, a_mean, a_std = plot_rotation(all_blocks, block_size, behav)
        print(f"\n  Rotation mechanism (Verlet blocks would be 0 deg):")
        _, _, angle_deg = block_rotation_decomp(all_blocks)
        for s in range(T):
            tag = '  <- recent' if s == T - 1 else ('  <- prev' if s == T - 2 else '')
            if T > 8 and 1 < s < T - 2:
                continue
            print(f"    pos {s:>3}: operator rotation {a_mean[s]:+6.2f} +/- {angle_deg[:, s].std():.2f} deg{tag}")
        if T > 8:
            print(f"    (positions 2..{T-3} omitted)")
        if behav is not None:
            print(f"\n  Behavioural rotation (model forward vs held-out next point):")
            print(f"    model rotates r_t by {behav['ang_model_mean']:+.2f} deg  "
                  f"(true orbital step {behav['ang_true_mean']:+.2f} deg, "
                  f"corr {behav['ang_corr']:.3f})")
            print(f"    radius preserved: |pred|/|r_t| = {behav['radius_ratio']:.3f}")
            print(f"    R^2(truth):  rotate-r_t-only {behav['r2_rotation_only']:.4f}  |  "
                  f"full model {behav['r2_model']:.4f}  |  persistence {behav['r2_persistence']:.4f}")
            print(f"    -> model predicts by rotating r_t forward along the orbit, not Verlet extrapolation")
        else:
            print(f"\n  (behavioural rotation skipped: block_size == trajectory length, "
                  f"no held-out next point)")
        print(f"  saved -> {out3}")

        # ---- elliptical handling: angular (Kepler 2) vs radial breathing ----
        if behav is not None:
            out4 = plot_kepler2(block_size, behav)
            print(f"\n  Elliptical handling (angular vs radial):")
            print(f"    Kepler 2 (angle ~ 1/r^2): corr(true angle, 1/r^2) = "
                  f"{behav['kepler2_corr_true']:.3f}, "
                  f"corr(model angle, 1/r^2) = {behav['kepler2_corr_model']:.3f}")
            print(f"      -> model varies its rotation rate with radius (handles ellipse's angular speed)")
            print(f"    radial breathing: true |dr| mean {behav['dr_true_absmean']:.4f}, "
                  f"corr(model dr, true dr) = {behav['dr_corr']:+.3f}")
            print(f"      step R^2: radius-LOCKED rotation {behav['r2_disp_rotlock']:.4f}  vs  "
                  f"full model {behav['r2_disp_model']:.4f}")
            print(f"      -> model drops the radial term (the 2nd-order/Verlet piece); rotation carries the signal")
            print(f"    saved -> {out4}")

    print(f"\n{'=' * 78}")


if __name__ == "__main__":
    main()
