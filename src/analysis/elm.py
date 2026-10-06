"""Equivalent Linear Mapping (ELM) analysis of the trained orbit models.

ELM freezes the model's input-dependent nonlinearities (LayerNorm scale, SiLU
gate, attention weights) at an operating point and detaches them, making the map
from embeddings to output *exactly* linear-homogeneous there. Differentiating
downstream of the positional-embedding sum and re-projecting through the input
embedding gives, for the last output token, the exact identity

    r_hat_{t+1} = Sum_s J_s @ r_s  +  P        (content blocks + positional term)

which is what everything here is built on. The reference test is whether a model
implements the Newtonian (Stormer-Verlet) one-step map,

    r_{t+1} = 2 r_t - r_{t-1} + a_t dt^2,     a_t = -GM r_t / |r_t|^3

i.e. constant-velocity extrapolation plus an O(dt^2) gravitational correction.
In ELM terms a Newtonian model has content blocks

    J_{recent = T-1}  ~  2 I      J_{prev = T-2}  ~ -1 I      J_{older}  ~  0

so the deviation of the measured blocks from (2I, -I) is the gravitational kick.

This module provides the measurement primitives (`collect_terms`,
`rotation_decompose`, `rotation_behavioral`, `rollout`) and the figure helpers
used by `index.ipynb`. It is a library, not a script -- the publication notebook
is the entry point.

The derivations behind these functions are written out in `decomposition_notes.md`.
"""

import io
import contextlib

import numpy as np
import torch

from .data import load_trajectories, sample_orbit_windows

torch.set_default_dtype(torch.float64)


def collect_terms(model, block_size, n_windows, seed=0, offset=0):
    """Per-token content Jacobians plus the size of every term in the ELM sum.

    The embedding-level decomposition writes the last-position prediction as one
    content vector per input token plus a single positional (wpe) vector:

        y* = Sum_t (J_{h,t} W_in) @ x_t  +  Sum_t J_{h,t} @ wpe[t]
             \-------- content --------/     \------ positional ------/

    Analyses that use only the content blocks J_{x,t} drop the positional term,
    which is safe when it is small against whatever is being measured and not
    otherwise.  This returns the terms themselves so the caller can check; it
    costs one Jacobian pass either way.

    Returns:
        all_blocks  (n, T, D, D)  per-token content Jacobians
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

# Arcadia palette. Kept here rather than taken from `arcadia_pycolor`, which the
# environment does not install -- see env.yml.
ARC_SLATE = '#403F4C'   # axes, text
ARC_VIBE  = '#5088C5'   # primary scatter
ARC_CORAL = '#F28360'   # fit line
ARC_GRID  = '#E8E8ED'   # light gridlines
ARC_DIAG  = '#8A8A99'   # y = x reference


def apply_figure_style():
    """Set the publication's matplotlib defaults, once, for every figure.

    rcParams is global and permanent, so this has to run before the first figure
    is drawn -- applying it partway through a notebook silently gives the figures
    above it one style and those below it another.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'font.size':          11,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.spines.left':   True,
        'axes.spines.bottom': True,
        'axes.edgecolor':     ARC_SLATE,
        'axes.linewidth':     0.8,
        'axes.labelcolor':    ARC_SLATE,
        'xtick.color':        ARC_SLATE,
        'ytick.color':        ARC_SLATE,
        'xtick.major.width':  0.8,
        'ytick.major.width':  0.8,
        'text.color':         ARC_SLATE,
        'figure.facecolor':   'white',
        'axes.facecolor':     'white',
    })


def plot_coefficient_plane(ax, coeffs):
    """The model's position in (c_{t-1}, c_t) space, one point per window.

    c = tr(J)/D is the isotropic part of a content Jacobian: the scalar weight the
    model places on that input position. Two references make the plot
    readable, and they say different things:

        (-1, 2)  inertial      constant-velocity extrapolation, 2 r_t - r_(t-1),
                               which is the linear part of a Verlet step
        the line c_(t-1) + c_t = 1   any weighted average of the two inputs

    A model doing forward extrapolation sits near (-1, 2); one doing interpolation
    sits on the dashed line. Drawn by both the noisy and noiseless figures, so the
    reference markers carry the same names in each.
    """
    T = coeffs.shape[1]
    c_prev, c_recent = coeffs[:, T - 2], coeffs[:, T - 1]

    ax.scatter(c_prev, c_recent, s=14, alpha=0.25, color='tab:blue',
               label='Per-window', zorder=2)
    ax.scatter([c_prev.mean()], [c_recent.mean()], s=130, marker='o',
               color='tab:blue', edgecolor='k', zorder=4,
               label=f'Measured mean')
    ax.scatter([-1], [2], s=130, marker='o', color='tab:red', edgecolor='k',
               label='Inertial point (-1, 2)', zorder=5)

    lo = min(-1.5, float(c_prev.min()))
    hi = max(2.5, float(c_recent.max()))
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, 1 - xs, ls='--', color='gray', lw=1, zorder=1,
            label=r'$c_{t-1}+c_t=1$')

    ax.set_xlabel(r'$c_{t-1}$  (previous position)')
    ax.set_ylabel(r'$c_t$  (most recent position)')
    ax.legend(loc='best')


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
    add to |y| and must not be stacked.
    """
    T = content.shape[1]
    nrm = lambda a: np.linalg.norm(a, axis=-1)

    series = [
        (r'$J_{t-1}\,r_{t-1}$',        nrm(content[:, T - 2])),
        (r'$J_t\,r_t$',                nrm(content[:, T - 1])),
        (r'$P$ (wpe)',                   nrm(positional)),
    ]
    for (label, mag), colour in zip(series, _TERM_COLOURS):
        ax.scatter(radius, mag, s=12, alpha=0.5, color=colour, linewidths=0,
                   label=f'{label}')

    ax.set_yscale('log')
    ax.set_xlabel(r'$|r_t|$')
    ax.set_ylabel('Magnitude of ELM term')
    ax.set_axisbelow(True)
    ax.legend(loc='lower right')


def kick_terms(content, positional, windows, dt):
    """The vector terms whose sum is the model's deviation from inertial coasting.

    Verlet's inertial part is 2 r_t - r_{t-1}; subtracting it from the ELM identity
    leaves exactly three terms, one per input token plus the positional one:

        r_hat - (2 r_t - r_{t-1}) = dJ_t @ r_t + dJ_{t-1} @ r_{t-1} + P
                                    \--------- content ---------/     wpe

    where dJ_t = J_t - 2I and dJ_{t-1} = J_{t-1} + I.  The ELM sum is a sum of
    VECTORS, so the dJ blocks are operators and only become contributions once they
    act on their inputs -- which is why the comparison has to be made on the
    products.  Those products need no matrix algebra: collect_terms() already
    returns J_{x,t} @ x_t as `content`, so dJ_t @ r_t is just content - 2 r_t.

    Args:
        content:    (n, T, D) per-token content contributions, from collect_terms()
        positional: (n, D)    positional (wpe) contribution, from collect_terms()
        windows:    (n, T, D) the input windows those terms were computed on
        dt:         sample spacing, used for the ideal Newtonian kick

    Returns a dict of (n, D) vectors -- `recent`, `prev`, `content`, `positional`,
    `total`, `grav` (the ideal kick -dt^2 r_t/|r_t|^3) -- plus `radius` (n,).
    """
    T = content.shape[1]
    r_prev, r_t = windows[:, T - 2, :], windows[:, T - 1, :]
    recent = content[:, T - 1] - 2.0 * r_t          # (J_recent - 2I) @ r_t
    prev = content[:, T - 2] + r_prev               # (J_prev   +  I) @ r_prev
    radius = np.linalg.norm(r_t, axis=1)
    return dict(recent=recent, prev=prev, content=recent + prev,
                positional=positional, total=recent + prev + positional,
                grav=-dt ** 2 * r_t / radius[:, None] ** 3, radius=radius)


def plot_kick_terms(axes, content, positional, windows, dt):
    """Three panels showing that the gravitational kick emerges only from the sum.

    Panels (a) and (b) take one component of the deviation from inertia each,
    resolved in that window's own gravity frame -- radial along the ideal kick (i.e.
    toward the star) and tangential perpendicular to it -- and plot the positional
    contribution against the content one, in units of |grav|.  Reproducing gravity
    means the radial pair sums to 1 and the tangential pair to 0, so each component
    has to land on a line of slope -1, drawn dashed.  Points are coloured by |r_t|.

    Note what carries the evidence in (a) and (b).  Where along its line a point
    sits is the result: it is how the labour is divided between the two terms, and
    many divisions reproduce the same total, so which one the network uses is not
    derivable from its behaviour.  That the lines sit at 1 and 0 is a consistency
    check -- the attribution route and the behavioural route reach the same total by
    different means.  The slope carries nothing: the two terms sum to a quantity far
    smaller than either, so a slope near -1 follows by construction.

    Panel (c) puts the radial division against radius directly, with medians over
    quantile bins and a bootstrap CI, since the trend is otherwise only implied by
    the colour scale in (a).  The bootstrap is paired across the two terms -- both
    are resampled on the same windows within a bin -- so the gap between the bands
    reflects the terms differing, not two independent resamplings differing.

    The line_deviation returned for (a) and (b) is identically |sum - target|: it
    measures how well the attributed TOTAL matches gravity, the same quantity
    @fig-newton-behavioral measures behaviourally, and says nothing about the split.
    It is returned rather than drawn, so the prose can quote it without the panel
    carrying a number the reader has to parse.

    Keeping the individual windows matters: a summary over windows cannot show that
    the cancellation holds window by window, and the tangential component in
    particular changes sign from window to window.

    Returns a dict with the fitted power law of each term's magnitude against r (not
    plotted -- @fig-newton-behavioral carries the log-log evidence -- so these are
    for checking the prose against the run), the median deviation from each
    constraint line, and the binned radial profile with its CI.
    """
    t = kick_terms(content, positional, windows, dt)
    g = np.linalg.norm(t['grav'], axis=1)
    ghat = t['grav'] / g[:, None]                        # unit vector toward the sun
    that = np.stack([-ghat[:, 1], ghat[:, 0]], axis=1)    # perpendicular to it
    radius = t['radius']
    vlo, vhi = np.percentile(radius, [2, 98])

    out = {'exponent': {}, 'line_deviation': {}, 'radial_profile': {}}
    for label, v in [('content', t['content']), ('positional', t['positional']),
                     ('sum', t['total']), ('ideal gravity', t['grav'])]:
        out['exponent'][label] = float(
            np.polyfit(np.log(radius), np.log(np.linalg.norm(v, axis=1)), 1)[0])

    series = [('content residual', t['content'], '#1f77b4'),
              ('positional (wpe)', t['positional'], '#9467bd'),
              ('their sum', t['total'], 'tab:green')]

    # --- (a), (b): the two components against their constraint lines --------------
    panels = [('radial', ghat, 1.0, '(a)'),
              ('tangential', that, 0.0, '(b)')]
    for ax, (name, proj, target, letter) in zip(axes[:2], panels):
        c = (t['content'] * proj).sum(1) / g
        p = (t['positional'] * proj).sum(1) / g
        out['line_deviation'][name] = float(np.median(np.abs(c + p - target)))

        lo, hi = np.percentile(np.concatenate([c, p]), [2, 98])
        pad = 0.15 * (hi - lo)
        lo, hi = min(lo - pad, -0.2), max(hi + pad, target + 0.2)
        span = np.linspace(lo, hi, 10)

        ax.plot(span, target - span, '--', color='0.25', lw=1.5,
                label=fr'{name}: sum $= {target:.0f}$  (Newtonian gravity)')
        sc = ax.scatter(c, p, s=20, alpha=0.8, c=radius, cmap='viridis',
                        vmin=vlo, vmax=vhi, linewidths=0)
        ax.axhline(0, color='0.88', lw=0.6); ax.axvline(0, color='0.88', lw=0.6)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
        ax.set_xlabel(r'content residual / $|\mathbf{g}|$')
        ax.set_ylabel(r'positional (wpe) / $|\mathbf{g}|$')
        ax.text(0.03, 0.97, letter, transform=ax.transAxes, va='top', ha='left',
                fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
    cb = axes[0].figure.colorbar(sc, ax=list(axes[:2]), fraction=0.046, pad=0.03)
    cb.set_label(r'$|\mathbf{r}_t|$')

    # --- (c): the radial division against radius ---------------------------------
    ax = axes[2]
    n_bins, n_boot = 5, 4000
    edges = np.percentile(radius, np.linspace(0, 100, n_bins + 1))

    # Bin once, and draw one bootstrap resampling per bin up front, from a single
    # stream so the bins are independent draws.  Every series is then resampled on
    # the SAME windows: the terms are measured per window, so pairing them removes
    # resampling noise from the comparison *between* series, which is what this panel
    # is for.  (Constructing the generator inside the loops instead would hand the
    # equal-sized quantile bins identical index matrices.)
    rg = np.random.default_rng(0)
    bins = []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (radius >= a) & (radius < b)
        if k.sum() < 8:
            continue
        n_k = int(k.sum())
        bins.append((k, rg.integers(0, n_k, size=(n_boot, n_k))))

    for label, v, colour in series:
        y = (v * ghat).sum(1) / g
        ax.scatter(radius, y, s=8, alpha=0.18, color=colour, linewidths=0)
        cen, med, blo, bhi = [], [], [], []
        for k, boot_idx in bins:
            yy = y[k]
            boot = np.median(yy[boot_idx], axis=1)
            cen.append(float(np.median(radius[k]))); med.append(float(np.median(yy)))
            q = np.percentile(boot, [2.5, 97.5]); blo.append(q[0]); bhi.append(q[1])
        ax.plot(cen, med, '-', color=colour, lw=2, ms=6, label=label)
        ax.fill_between(cen, blo, bhi, color=colour, alpha=0.22)
        out['radial_profile'][label] = dict(r=cen, median=med, lo=blo, hi=bhi)
    ax.axhline(1, color='k', ls='--', lw=1.2)
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_ylim(-1.6, 3.0)
    ax.set_xlabel(r'$|\mathbf{r}_t|$')
    ax.set_ylabel(r'radial component / $|g|$')
    ax.text(0.97, 0.97, '(c)', transform=ax.transAxes, va='top', ha='right')
    ax.legend(loc='upper left')
    return out


def residual_structure(all_blocks, windows):
    """Invariant structure of the two deviation-from-inertia blocks.

    dJ_t = J_t - 2I and dJ_(t-1) = J_(t-1) + I.  Each 2x2 block decomposes into
    four orthogonal parts under the Frobenius inner product -- isotropic d,
    rotation generator a, and two shears -- of which d, a and the rotation score
    a^2/(a^2 + shear^2) are invariant under rotation of coordinates, so they
    describe the operators without reference to a choice of axes.

    Returns a dict of per-window arrays plus the rotation score an unstructured
    2x2 matrix would give, as a reference point for the score.
    """
    I2 = np.eye(2)
    T = all_blocks.shape[1]
    dJr = all_blocks[:, T - 1] - 2 * I2
    dJp = all_blocks[:, T - 2] + I2
    radius = np.linalg.norm(windows[:, T - 1, :], axis=1)
    fro = lambda M: np.linalg.norm(M, axis=(1, 2))

    def parts(M):
        # Section-A decomposition; the residual blocks are small, so the score's
        # guard epsilon goes well below the scale of their shears.
        d, a, _, score = rotation_decompose(M, eps=1e-30)
        return d, a, score

    d_r, a_r, s_r = parts(dJr)
    d_p, a_p, s_p = parts(dJp)
    rng = np.random.default_rng(0)
    return dict(dJr=dJr, dJp=dJp, radius=radius,
                d_r=d_r, a_r=a_r, score_r=s_r, d_p=d_p, a_p=a_p, score_p=s_p,
                fail=fro(dJp + dJr) / fro(dJr),
                r2=float(np.corrcoef(dJp.ravel(), (-dJr).ravel())[0, 1] ** 2),
                score_random=float(np.median(parts(rng.normal(size=(20000, 2, 2)))[2])))


def _mirror_scatter(ax, cx, cy, radius, sym, title, vlo, vhi, pct=99, pad=0.10):
    """Previous-block coefficient against recent-block, with the y = -x reference.

    Square box snug to the data with equal aspect, so the -45 deg
    (equal-and-opposite) line reads true.  The annotation gives the fitted slope
    and the median per-window departure, which is what distinguishes the two
    coefficients: independent auto-scaling would otherwise make both look like
    clean anti-diagonals.
    """
    x0, x1 = np.percentile(cx, [100 - pct, pct]); y0, y1 = np.percentile(cy, [100 - pct, pct])
    x0, x1 = min(x0, 0), max(x1, 0);              y0, y1 = min(y0, 0), max(y1, 0)
    side = max(x1 - x0, y1 - y0) * (1 + pad)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    xL, xR, yB, yT = mx - side / 2, mx + side / 2, my - side / 2, my + side / 2

    ax.axhline(0, color='0.85', lw=0.6); ax.axvline(0, color='0.85', lw=0.6)
    span = np.linspace(min(xL, yB), max(xR, yT), 10)
    ax.plot(span, -span, 'k--', lw=1.2, label='equal-and-opposite')
    sc = ax.scatter(cx, cy, s=14, alpha=0.6, c=radius, cmap='viridis',
                    vmin=vlo, vmax=vhi, linewidths=0)
    ax.set_xlim(xL, xR); ax.set_ylim(yB, yT); ax.set_aspect('equal')
    ax.set_xlabel(fr'${sym}_t$  (recent block)')
    ax.set_ylabel(fr'${sym}_{{t-1}}$  (previous block)')
    ax.set_title(title, fontsize=11)
    slope = np.polyfit(cx, cy, 1)[0]
    dep = np.median(np.abs(cx + cy) / np.abs(cx))
    ax.text(0.04, 0.04, f'slope {slope:+.2f}\ndeparture {dep:.0%}',
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
    ax.legend(fontsize=8, loc='upper right')
    return sc, float(slope), float(dep)


def plot_residual_structure(axes, all_blocks, windows):
    """Four panels on the two deviation-from-inertia operators (2x2 axes grid).

    [a] every matrix entry of the previous block against the negative of the
        corresponding recent-block entry -- the negation, entry by entry.
    [b] the rotation score of each block, against the value an unstructured 2x2
        matrix would give: the blocks are scaled rotations, shear negligible.
    [c] the rotation generator a, previous against recent.
    [d] the isotropic coefficient d, previous against recent.

    (c) and (d) share a construction but not a conclusion: a is
    equal-and-opposite across the pair while d is only about half so, which is
    why each panel reports its fitted slope rather than relying on the eye --
    the snug per-panel scaling makes both clouds look like anti-diagonals.

    These describe the operators.  They do not describe what the output depends
    on: dJ_(t-1) acts on r_(t-1), which is several times longer than the step
    r_t - r_(t-1), so the small residual left by the imperfect negation
    contributes to the correction on a par with the difference term.

    Returns the dict from residual_structure(), with the fitted slopes added.
    """
    st = residual_structure(all_blocks, windows)
    radius = st['radius']
    vlo, vhi = np.percentile(radius, [2, 98])

    # --- (a) entry-wise negation -------------------------------------------------
    ax = axes[0, 0]
    x = (-st['dJr']).reshape(-1); y = st['dJp'].reshape(-1)
    lim = 1.05 * max(np.abs(x).max(), np.abs(y).max())
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1.3, label=r'$\mathrm{d}J_{t-1} = -\mathrm{d}J_t$')
    sc = ax.scatter(x, y, s=7, alpha=0.5, c=np.repeat(radius, 4), cmap='viridis',
                    vmin=vlo, vmax=vhi, linewidths=0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.axhline(0, color='0.85', lw=0.5); ax.axvline(0, color='0.85', lw=0.5)
    ax.set_xlabel(r'$-\,(\mathrm{d}J_t)_{ij}$'); ax.set_ylabel(r'$(\mathrm{d}J_{t-1})_{ij}$')
    ax.set_title('(a) Every entry: previous $= -$ recent', fontsize=11)
    ax.text(0.04, 0.88, fr"$R^2 = {st['r2']:.3f}$", transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.85))
    ax.legend(fontsize=8, loc='lower right')

    # --- (b) the blocks are scaled rotations -------------------------------------
    # the scores all exceed ~0.97, so a [0, 1] axis collapses them into one bin
    ax = axes[0, 1]
    xlo = min(0.95, float(np.percentile(np.r_[st['score_r'], st['score_p']], 0.5)))
    for key, colour, lab in [('score_r', 'tab:purple', 'recent'),
                             ('score_p', 'tab:orange', 'previous')]:
        ax.hist(st[key], bins=40, range=(xlo, 1.0), color=colour, alpha=0.55,
                label=f"{lab} (median {np.median(st[key]):.3f})")
    ax.axvline(1, color='k', ls='--', lw=1, label='pure rotation')
    ax.set_xlim(xlo, 1.0 + 0.02 * (1.0 - xlo))
    ax.annotate(f"an unstructured $2\\times2$ matrix\nwould sit at {st['score_random']:.2f},\n"
                "far off-scale to the left",
                xy=(0.03, 0.60), xycoords='axes fraction', fontsize=7.5, color='tab:red')
    ax.set_xlabel(r'rotation score  $a^2/(a^2+\mathrm{shear}^2)$')
    ax.set_ylabel('windows')
    ax.set_title('(b) Each block is a near-pure scaled rotation', fontsize=11)
    ax.legend(fontsize=7.5, loc='upper left')

    # --- (c), (d) the two invariant coefficients, one panel each ------------------
    sc, st['slope_a'], st['dep_a'] = _mirror_scatter(
        axes[1, 0], st['a_r'], st['a_p'], radius, 'a',
        '(c) Rotation generators are equal-and-opposite', vlo, vhi)
    sc, st['slope_d'], st['dep_d'] = _mirror_scatter(
        axes[1, 1], st['d_r'], st['d_p'], radius, 'c',
        '(d) Isotropic coefficients only about half so', vlo, vhi)

    cb = axes[0, 0].figure.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.030, pad=0.02)
    cb.set_label(r'$|\mathbf{r}_t|$')
    return st


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


def rollout(model, seed, n_steps):
    """Autoregressive rollout from a seed prefix, sliding a block_size window.

    `seed` is a (k, D) array of k >= 1 initial positions.  Each step feeds the model
    the last block_size points available (all of them while fewer than block_size
    have accumulated) and appends the prediction -- the same sliding-window
    generation Liu et al. use, so models of different context length can be rolled
    out from an identical seed and compared.  Returns the (k + n_steps, D) trajectory.

    For a block_size=2 model this is exactly "feed the last two points".
    """
    bs = model.config.block_size
    dt = next(model.parameters()).dtype
    traj = [np.asarray(p, float) for p in np.asarray(seed)]
    for _ in range(n_steps):
        win = torch.tensor(np.stack(traj[-bs:]), dtype=dt).unsqueeze(0)
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


def rotation_decompose(blocks, eps=1e-12):
    """Decompose 2x2 matrices into scale, rotation, and a rotation-score.

    This is the section-A decomposition of `decomposition_notes.md`, and the one
    place it is implemented: any 2x2 matrix splits into four orthogonal parts
    under the Frobenius inner product -- isotropic, rotation, and two shears.

    Accepts a single (2, 2) matrix or a stack (..., 2, 2) and returns
    (scale, anti, angle_deg, rot_score), where
        scale     = (J00 + J11) / 2                 isotropic magnitude
        anti      = (J10 - J01) / 2                 antisymmetric (rotation) part
        angle_deg = atan2(anti, scale)              implied rotation angle
        rot_score = anti^2 / (anti^2 + shear^2 + diag_diff^2)   (1 = pure rotation)
    so a matrix with J ~ scale * R(angle) has rot_score ~ 1.

    `eps` guards the rot_score denominator against an exactly-zero block; it is
    exposed only so callers working on small residual blocks can push it below
    the scale of their own shears.
    """
    J = np.asarray(blocks)
    ds     = (J[..., 0, 0] + J[..., 1, 1]) / 2
    anti   = (J[..., 1, 0] - J[..., 0, 1]) / 2
    sym_od = (J[..., 0, 1] + J[..., 1, 0]) / 2
    diag_d = (J[..., 0, 0] - J[..., 1, 1]) / 2
    angle_deg = np.degrees(np.arctan2(anti, ds))
    rot_score = anti**2 / (anti**2 + sym_od**2 + diag_d**2 + eps)
    return ds, anti, angle_deg, rot_score


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

    # Specific angular momentum TIMES the sample spacing, L_z * dt.  The data
    # carries positions only, so there is no velocity to cross with r_t; we use
    # the in-window backward displacement d = r_t - r_{t-1} = v dt + O(dt^2),
    # making r_t x d a first-order estimate of L_z * dt, not of L_z itself.
    # That is the natural quantity here anyway: Kepler's 2nd law in per-step
    # form reads dtheta = (L_z dt) / r^2, so L_z * dt is exactly the numerator
    # of the per-step angle -- the hidden axis behind the dtheta-vs-1/r^2
    # scatter (constant-L orbits trace constant-slope rays).
    disp = r_t - windows[:, -2, :]
    Lz_dt = r_t[:, 0] * disp[:, 1] - r_t[:, 1] * disp[:, 0]

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
        'Lz_dt': Lz_dt,     # = L_z * dt (first order); see comment above
        'kepler2_corr_true': float(np.corrcoef(ang_true, inv_r2)[0, 1]),
        'kepler2_corr_model': float(np.corrcoef(ang_model, inv_r2)[0, 1]),
        'dr_corr': float(np.corrcoef(dr_model, dr_true)[0, 1]),
        'dr_true_absmean': float(np.abs(dr_true).mean()),
        'r2_disp_rotlock': r2(rot_only - r_t, d),   # radius-LOCKED rotation
        'r2_disp_model': r2(pred - r_t, d),         # full model
        'n': int(n_windows),
    }
