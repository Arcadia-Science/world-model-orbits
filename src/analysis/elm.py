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


def collect_terms(model, block_size, n_windows, seed=0, offset=0):
    """Like collect(), but also returns the size of every term in the ELM sum.

    The embedding-level decomposition writes the last-position prediction as one
    content vector per input token plus a single positional (wpe) vector:

        y* = Sum_t (J_{h,t} W_in) @ x_t  +  Sum_t J_{h,t} @ wpe[t]
             \-------- content --------/     \------ positional ------/

    collect() exposes only the content blocks J_{x,t}; the positional term is
    dropped, which is safe when it is small against whatever is being measured
    and not otherwise.  This returns the terms themselves so the caller can
    check.  One Jacobian pass, so it costs the same as collect().

    Returns:
        all_blocks  (n, T, D, D)  per-token content Jacobians (as collect())
        coeffs      (n, T)        isotropic coefficients tr(J_s)/D
        content     (n, T, D)     per-token content contribution J_{x,t} @ x_t
        positional  (n, D)        positional contribution, summed over tokens
        radius      (n,)          |r_t| of the most recent input position
        recon_err_max             worst-case relative reconstruction error
    """
    windows = sample_orbit_windows(block_size, n_windows, seed=seed, offset=offset)
    all_blocks, content, positional, recon_errs = [], [], [], []

    for w in windows:
        x_star = torch.tensor(w, dtype=torch.float64).unsqueeze(0)
        T = x_star.shape[1]
        D = x_star.shape[2]
        last = slice((T - 1) * D, T * D)

        (jac_h, jac_x, _, recon_err, _, _, _, _) = \
            model.compute_detached_jacobians_embedding(x_star)

        pos     = torch.arange(0, T, dtype=torch.long, device=x_star.device)
        pos_emb = model.transformer.wpe(pos)

        blocks = torch.stack([jac_x[t][last].detach() for t in range(T)])   # (T, D, D)
        per_token = torch.stack([
            jac_x[t][last].detach() @ x_star[0, t, :] for t in range(T)
        ])                                                                  # (T, D)
        pos_vec = sum(jac_h[t][last].detach() @ pos_emb[t] for t in range(T))

        all_blocks.append(blocks.numpy())
        content.append(per_token.numpy())
        positional.append(pos_vec.detach().numpy())
        recon_errs.append(recon_err)

    all_blocks = np.array(all_blocks)                     # (n, T, D, D)
    content    = np.array(content)                        # (n, T, D)
    positional = np.array(positional)                     # (n, D)
    D = all_blocks.shape[-1]
    coeffs = np.trace(all_blocks, axis1=2, axis2=3) / D   # (n, T)
    radius = np.linalg.norm(windows[:, -1, :], axis=1)    # (n,)
    return all_blocks, coeffs, content, positional, radius, float(np.max(recon_errs))


# Fixed hue order for the three ELM terms -- assigned by identity, never cycled.
# Checked for colourblind separation (worst adjacent pair dE 24.6 protan).
_TERM_COLOURS = ('#ff7f0e', '#1f77b4', '#9467bd')     # prev, recent, positional


def plot_term_magnitudes(ax, content, positional, radius):
    """Size of every term in the ELM sum, per window, against |r_t|.

    Draws the two most recent content terms plus the positional term (for
    block_size 2 that is the whole sum; older tokens are not drawn).

    Args:
        content:    (n, T, D) per-token content contributions, from collect_terms()
        positional: (n, D)    positional (wpe) contribution
        radius:     (n,)      |r_t|

    Log y, because the positional term runs a couple of percent of the
    prediction and is invisible on a linear axis beside the content terms.

    The terms are vectors summing to the prediction, so their magnitudes do NOT
    add to |y| and must not be stacked.  The annotation reports
    Sum_terms |term| / |y|, which is 1 when the terms simply add and grows as
    they cancel -- a forward extrapolator builds a small prediction out of
    large opposing terms, a weighted average does not.
    """
    T = content.shape[1]
    nrm = lambda a: np.linalg.norm(a, axis=-1)
    y_mag = nrm(content.sum(1) + positional)

    series = [
        (r'$J_{\rm prev}\,\mathbf{r}_{t-1}$',   nrm(content[:, T - 2])),
        (r'$J_{\rm recent}\,\mathbf{r}_{t}$',   nrm(content[:, T - 1])),
        (r'positional (wpe)',                   nrm(positional)),
    ]
    for (label, mag), colour in zip(series, _TERM_COLOURS):
        ax.scatter(radius, mag, s=12, alpha=0.55, color=colour, linewidths=0,
                   label=f'{label}   median {np.median(mag):.3f}')

    cancellation = np.median(sum(mag for _, mag in series) / y_mag)
    ax.text(0.03, 0.03,
            r'$\sum|{\rm term}|\,/\,|y|$ = ' + f'{cancellation:.2f}'
            + '\n(1.0 = terms add, >1 = they cancel)',
            transform=ax.transAxes, fontsize=8, va='bottom',
            bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))

    ax.set_yscale('log')
    ax.set_xlabel(r'$|\mathbf{r}_t|$')
    ax.set_ylabel('magnitude of ELM term')
    ax.grid(True, which='both', alpha=0.2, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc='center right')


def reconstruction_scatter(model, block_size, n_windows, seed=0, offset=0):
    """ELM reconstruction check for the LAST output token, over sampled windows.

    For each window this compares the model's forward output to the embedding-level
    ELM reconstruction  (content + positional = Sum_t J_{h,t} @ h_t).  Returns
    (model_values, elm_values), each flattened over windows, for a y=x scatter:
    a tight identity line (relative error ~1e-10) confirms the detached Jacobian is
    an exact linear stand-in for the model.
    """
    windows = sample_orbit_windows(block_size, n_windows, seed=seed, offset=offset)
    model_vals, emb_vals = [], []

    for w in windows:
        xs = torch.tensor(w, dtype=torch.float64).unsqueeze(0)

        with torch.no_grad():
            y_model, _ = model(xs)
        y_last = y_model[0, -1, :].detach().numpy()

        T   = xs.shape[1]
        pos = torch.arange(0, T, dtype=torch.long)
        x_emb   = model.embed_continuous(xs)
        pos_emb = model.transformer.wpe(pos)
        h_star  = (x_emb + pos_emb).detach()

        model.detach_for_elm = True
        model.eval()

        def last_output(h_in):
            out = model.forward_from_embedding(h_in)
            return out[0, -1, :]

        content_last    = torch.zeros(2, dtype=torch.float64)
        positional_last = torch.zeros(2, dtype=torch.float64)

        for t in range(T):
            before = h_star[0, :t, :].detach()
            after  = h_star[0, t+1:, :].detach()

            def fwd(h_t, before=before, after=after):
                h_in = torch.cat([before, h_t.unsqueeze(0), after], dim=0).unsqueeze(0)
                with torch.enable_grad():
                    return last_output(h_in)

            h_t = h_star[0, t, :].detach().requires_grad_(True)
            J_ht = torch.autograd.functional.jacobian(
                fwd, h_t, create_graph=False, strict=False
            )

            W_in  = model.input_embedding.weight
            J_xt  = J_ht @ W_in
            content_last    += J_xt @ xs[0, t, :]
            positional_last += J_ht @ pos_emb[t]

        model.detach_for_elm = False

        recon = (content_last + positional_last).detach().numpy()
        model_vals.append(y_last)
        emb_vals.append(recon)

    return np.concatenate(model_vals), np.concatenate(emb_vals)


def rotation_channel_decomposition(model, block_size, n_windows, seed=0, offset=0):
    """Attribute the model's per-step turn to isotropic / generator / shear / positional channels.

    For each real-orbit window (intended for block_size == 2) the last-token prediction is
    the EXACT ELM sum of four contributions applied to the two input positions r_{t-1}, r_t:

        r_hat_{t+1} = iso + generator + shear + positional      (content = iso + gen + shear)

    In 2D a scaled rotation is a complex number: writing r = x + iy, a content block splits as
        J @ r  =  s*r  +  a*(i r)  +  shear,      s = tr(J)/2,  a = (J_10 - J_01)/2,
    i.e. a *collinear* (radial) isotropic part s*r, a *perpendicular* (tangential) generator
    part a*(i r), and a symmetric-traceless shear remainder. s is exactly the isotropic
    coefficient c_s = tr(J)/D; the generator a carries the turn (theta = atan2(a, s)).

    Each channel vector is projected onto the local frame of r_t (radial rhat = r_t/|r_t|,
    tangential = i*rhat) and normalised by |r_t|. The *tangential* share is that channel's
    contribution to the per-step turn (rotation = tangential displacement per radius); the
    *radial* share is its contribution to the radius change. Combining the two tokens is
    just complex addition, so the isotropic coefficient and the generator enter the same
    sum -- no separate handling of c_s is needed.

    Uses the same sampled windows as `collect` (matching seed/offset), so the returned
    per-window arrays line up with the Jacobian blocks analysed elsewhere.

    Returns a dict of per-window arrays plus summary scalars:
        tang_{iso,gen,shear,pos} : tangential share of each channel (fraction of |r_t|)
        rad_{iso,gen,shear,pos}  : radial share of each channel
        dth_beh                  : behavioural turn, signed angle(r_t -> model pred) [deg]
        dth_rot                  : turn of the rotation-only reconstruction iso+gen [deg]
        radius_ratio             : |model pred| / |r_t|
        corr_rot_beh             : corr(dth_rot, dth_beh) over windows
        recon_err_max            : max ||(iso+gen+shear+pos) - model output|| (ELM closure)
    """
    windows = sample_orbit_windows(block_size, n_windows, seed=seed, offset=offset)
    T, D = block_size, windows.shape[-1]
    last = slice((T - 1) * D, T * D)
    mdtype = next(model.parameters()).dtype

    def cx(v):                                        # (..., 2) real -> complex
        return v[..., 0] + 1j * v[..., 1]

    keys = ['iso', 'gen', 'shear', 'pos']
    out = {f'tang_{k}': [] for k in keys}
    out.update({f'rad_{k}': [] for k in keys})
    out.update(dth_beh=[], dth_rot=[], radius_ratio=[], recon_err=[])

    for w in windows:
        x = torch.tensor(w, dtype=torch.float64).unsqueeze(0)
        _, jac_x, _, _, _, pos_c, _, _ = model.compute_detached_jacobians_embedding(x)
        Jr = jac_x[T - 1][last].detach().numpy()      # content block, recent -> last output
        Jp = jac_x[T - 2][last].detach().numpy()      # content block, prev   -> last output
        pos_vec = pos_c[last].detach().numpy()        # positional contribution to last output

        r_t, r_p = w[T - 1], w[T - 2]
        zt, zp = cx(r_t), cx(r_p)

        def parts(J):
            return (J[0, 0] + J[1, 1]) / 2, (J[1, 0] - J[0, 1]) / 2   # (isotropic s, generator a)
        sr, ar = parts(Jr)
        sp, ap = parts(Jp)

        z_iso   = sr * zt + sp * zp
        z_gen   = 1j * (ar * zt + ap * zp)
        z_cont  = cx(Jr @ r_t) + cx(Jp @ r_p)         # full content (iso + gen + shear)
        z_shear = z_cont - z_iso - z_gen
        z_pos   = cx(pos_vec)

        with torch.no_grad():
            z_pred = cx(model(x.to(mdtype))[0][0, -1, :].double().numpy())
        out['recon_err'].append(abs((z_iso + z_gen + z_shear + z_pos) - z_pred))

        rhat = zt / abs(zt)
        for k, v in zip(keys, [z_iso, z_gen, z_shear, z_pos]):
            proj = v * np.conj(rhat)                  # rotate channel into r_t's frame
            out[f'rad_{k}'].append(proj.real / abs(zt))   # component along r_t
            out[f'tang_{k}'].append(proj.imag / abs(zt))  # component along i*r_t (CCW forward)

        out['dth_beh'].append(np.degrees(np.angle(z_pred / zt)))
        out['dth_rot'].append(np.degrees(np.angle((z_iso + z_gen) / zt)))
        out['radius_ratio'].append(abs(z_pred) / abs(zt))

    out = {k: np.asarray(v) for k, v in out.items()}
    out['corr_rot_beh'] = float(np.corrcoef(out['dth_rot'], out['dth_beh'])[0, 1])
    out['recon_err_max'] = float(out.pop('recon_err').max())
    return out


def rollout(model, seed, n_steps):
    """Autoregressive rollout of a block_size=2 model from two seed positions.

    `seed` is a (2, 2) array of the first two positions; feeds the model its own
    predictions and returns the (n_steps+2, 2) generated trajectory.
    """
    dt = next(model.parameters()).dtype
    traj = [np.asarray(seed[0], float), np.asarray(seed[1], float)]
    for _ in range(n_steps):
        win = torch.tensor(np.stack(traj[-2:]), dtype=dt).unsqueeze(0)
        with torch.no_grad():
            traj.append(model(win)[0][0, -1].double().numpy())
    return np.array(traj)


def rollout_verlet(seed, n_steps, dt=0.2):
    """Ideal Stormer-Verlet rollout under inverse-square gravity (GM=1)."""
    traj = [np.asarray(seed[0], float), np.asarray(seed[1], float)]
    for _ in range(n_steps):
        r_t, r_pr = traj[-1], traj[-2]
        traj.append(2 * r_t - r_pr - dt**2 * r_t / np.linalg.norm(r_t)**3)
    return np.array(traj)


def gravity_kick_frames(model, trajectory, dt=0.2):
    """Per-step decomposition of the Verlet-deviation gravity kick along one orbit.

    For each interior step t (using the window r_{t-1}, r_t), returns the two input
    positions, the Verlet (inertial) prediction, the ideal Newtonian gravity vector, and
    the four channel vectors -- isotropic, rotation, shear, positional (wpe) -- whose sum
    is the model's non-inertial correction (pred - Verlet). Returns a list of dicts,
    one per step; intended to drive an animation of the decomposition.
    """
    I2 = np.eye(2)
    Rm = np.array([[0., -1.], [1., 0.]])
    frames = []
    for t in range(1, len(trajectory) - 1):
        r_pr, r_t = trajectory[t - 1], trajectory[t]
        xs = torch.tensor(np.stack([r_pr, r_t]), dtype=torch.float64).unsqueeze(0)
        _, jac_x, _, _, _, pos_c, _, _ = model.compute_detached_jacobians_embedding(xs)
        last = slice(2, 4)                       # last output token, D=2
        Jrec = jac_x[1][last].detach().numpy()
        Jprv = jac_x[0][last].detach().numpy()
        wpe = pos_c[last].detach().numpy()

        def channels(dJ):
            ds = (dJ[0, 0] + dJ[1, 1]) / 2
            a = (dJ[1, 0] - dJ[0, 1]) / 2
            sym = (dJ + dJ.T) / 2
            G = sym - ((sym[0, 0] + sym[1, 1]) / 2) * I2
            return ds, a, G

        ds_r, a_r, G_r = channels(Jrec - 2 * I2)
        ds_p, a_p, G_p = channels(Jprv + I2)
        iso = ds_r * r_t + ds_p * r_pr
        rot = (a_r * r_t + a_p * r_pr) @ Rm.T
        shear = G_r @ r_t + G_p @ r_pr
        frames.append(dict(
            r_pr=r_pr, r_t=r_t,
            verlet=2 * r_t - r_pr,
            grav=-dt**2 * r_t / np.linalg.norm(r_t)**3,
            iso=iso, rot=rot, shear=shear, wpe=wpe,
        ))
    return frames


def rotation_decompose(blocks):
    """Decompose 2x2 matrices into scale, rotation, and a rotation-score.

    Accepts a single (2, 2) matrix or a stack (..., 2, 2) and returns
    (scale, anti, angle_deg, rot_score), where
        scale     = (J00 + J11) / 2                 isotropic magnitude
        anti      = (J10 - J01) / 2                 antisymmetric (rotation) part
        angle_deg = atan2(anti, scale)              implied rotation angle
        rot_score = anti^2 / (anti^2 + shear^2 + diag_diff^2)   (1 = pure rotation)
    so a matrix with J ~ scale * R(angle) has rot_score ~ 1.
    """
    J = np.asarray(blocks)
    ds     = (J[..., 0, 0] + J[..., 1, 1]) / 2
    anti   = (J[..., 1, 0] - J[..., 0, 1]) / 2
    sym_od = (J[..., 0, 1] + J[..., 1, 0]) / 2
    diag_d = (J[..., 0, 0] - J[..., 1, 1]) / 2
    angle_deg = np.degrees(np.arctan2(anti, ds))
    rot_score = anti**2 / (anti**2 + sym_od**2 + diag_d**2 + 1e-12)
    return ds, anti, angle_deg, rot_score


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
