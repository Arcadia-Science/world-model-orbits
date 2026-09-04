# Matrix-decomposition notes (for verification)

These notes lay out, in full, the two decompositions used in the mechanistic
analysis of the **noiseless** `block_size = 2` model, plus the Verlet-inertia
residual that ties them to physics. They are written to be checked against the
code: every formula is annotated with the file/function and, where useful, the
`(x, y)` index convention that the implementation uses.

The intent is that a reader can (a) re-derive each identity by hand and (b) map
it onto the exact array operations in the source.

---

## 0. Setup, notation, conventions

**Task.** A GPT-2-style transformer reads a window of past 2-D positions and
predicts the next one. For the mechanistic work we use `block_size = 2`, so the
input window is the pair of positions $(\mathbf r_{t-1}, \mathbf r_t)$ and the
model outputs a prediction $\hat{\mathbf r}_{t+1}$ for the last token.

**ELM content blocks.** Equivalent Linear Mapping freezes the model's
input-dependent nonlinearities (LayerNorm scale, SiLU gate, attention weights)
at an operating point $x$ and detaches them, so the map from embeddings to
output is *exactly* linear-homogeneous at $x$. Differentiating downstream of
the positional-embedding sum and re-projecting through the input embedding
$W_{\rm in}$ gives, for the last output token, the exact identity

$$
\hat{\mathbf r}_{t+1}
= \underbrace{\sum_{s} J_s\,\mathbf r_s}_{\text{content}}
+ \underbrace{\sum_s J_{h,s}\,\mathrm{wpe}[s]}_{\text{positional}},
\qquad J_s \equiv J^{\rm last}_{h,s}\,W_{\rm in}\in\mathbb R^{2\times2}.
$$

The reconstruction error of this identity is $\sim 10^{-9}$ (float64), so each
per-token content block $J_s$ is a trustworthy stand-in for the model at that
window. For `block_size = 2` there are two content blocks:

- $J_{\rm recent} \equiv J_t$   — how the most recent position $\mathbf r_t$ drives the prediction   (`all_blocks[:, T-1]`)
- $J_{\rm prev} \equiv J_{t-1}$ — how the previous position $\mathbf r_{t-1}$ drives it   (`all_blocks[:, T-2]`)

**Index convention.** Everywhere in the code, for a block $J$:

- index `0` = the $x$ component, index `1` = the $y$ component;
- $J_{ij} = \partial(\text{output}_i)/\partial(\text{input}_j)$;
- `D = 2`, `dt = 0.2`, $GM = 1$, `torch.set_default_dtype(torch.float64)`.

**Code entry points.**
- `analysis.elm.collect_terms` — samples $n$ real-orbit windows, returns
  `all_blocks (n, T, 2, 2)` and `coeffs (n, T)` where `coeffs = trace(J)/D`,
  alongside the content, positional and radius terms of the ELM sum.
- `analysis.elm.rotation_decompose` — the §A coefficients. Accepts a single
  `(2, 2)` block or a stack, so it covers both the averaged and per-window cases.
- `analysis.elm.residual_structure` / `plot_residual_structure` — the §B
  (common/difference) analysis behind `index.ipynb`'s `fig-residual-structure`.

---

## A. Isotropic / rotation / shear decomposition of a 2×2 block

Any real $2\times2$ matrix decomposes into **four** mutually orthogonal
components (orthogonal under the Frobenius inner product
$\langle A,B\rangle = \operatorname{tr}(A^\top B)$). Using the basis matrices

$$
I=\begin{pmatrix}1&0\\0&1\end{pmatrix},\quad
R=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\quad
E_1=\begin{pmatrix}1&0\\0&-1\end{pmatrix},\quad
E_2=\begin{pmatrix}0&1\\1&0\end{pmatrix},
$$

we write

$$
J \;=\; \underbrace{s\,I}_{\text{isotropic}}
\;+\; \underbrace{a\,R}_{\text{rotation}}
\;+\; \underbrace{\gamma_{\rm d}\,E_1 + \gamma_{\rm o}\,E_2}_{\text{shear (symmetric-traceless)}},
$$

with the coefficients read straight off the entries:

$$
\boxed{\;
s = \tfrac12(J_{00}+J_{11}),\quad
a = \tfrac12(J_{10}-J_{01}),\quad
\gamma_{\rm d} = \tfrac12(J_{00}-J_{11}),\quad
\gamma_{\rm o} = \tfrac12(J_{01}+J_{10}).
\;}
$$

- **$s$ — isotropic scale.** Equal to $\operatorname{tr}(J)/2$, i.e. exactly the
  coefficient `c_s = trace(J_s)/D` used in the $(c_{\rm prev}, c_{\rm recent})$
  plots. A uniform radial gain: maps $\mathbf r \mapsto s\,\mathbf r$.
- **$a$ — rotation generator** (antisymmetric part). $R$ is the $90^\circ$
  rotation, so $a\,R$ turns its input by $+90^\circ$ and scales by $a$. Together
  with $s$ it forms a *scaled rotation*: $sI + aR$ has magnitude
  $\sqrt{s^2+a^2}$ and angle $\theta = \operatorname{atan2}(a, s)$.
- **$\gamma_{\rm d}, \gamma_{\rm o}$ — shear** (the symmetric-traceless remainder):
  direction-changing distortion that is *not* a rotation (unequal diagonal
  stretch $\gamma_{\rm d}$; $45^\circ$-axis stretch $\gamma_{\rm o}$).

Because the four basis matrices are orthogonal and each has Frobenius norm
$\sqrt2$,

$$
\|J\|_F^2 = 2\big(s^2 + a^2 + \gamma_{\rm d}^2 + \gamma_{\rm o}^2\big),
$$

so $s^2, a^2, \gamma_{\rm d}^2, \gamma_{\rm o}^2$ are an exact energy budget of the block.

### Rotation-score

To ask *"is the direction-changing action a turn or a distortion?"* we compare
the rotation energy against the shear energy, **deliberately excluding** the
isotropic $s$ (which changes size, not direction):

$$
\boxed{\;\text{rot-score} \;=\; \frac{a^2}{a^2 + \gamma_{\rm d}^2 + \gamma_{\rm o}^2}\;}
\qquad\in[0,1],\quad 1 \Leftrightarrow \text{pure scaled rotation}.
$$

A score $\approx 1$ means shear $\approx 0$, i.e. $J \approx s\,I + a\,R = \|{\cdot}\|\,R(\theta)$.

### Complex-number view (how the code combines the two tokens)

Identifying $\mathbf r = (x,y)\leftrightarrow z = x + iy$, the scaled-rotation
action of a block is complex multiplication:

$$
(sI + aR)\,\mathbf r \;\leftrightarrow\; s\,z + a\,(i\,z),
$$

since $i\,z$ is the $90^\circ$ CCW rotation of $z$. This is why the two tokens'
isotropic and generator parts add as plain complex numbers — no special handling
of $s$ vs $a$:

```
z_iso   = s_r * z_t + s_p * z_{t-1}
z_gen   = 1j * (a_r * z_t + a_p * z_{t-1})
z_shear = (full content) - z_iso - z_gen        # the remainder
```

### Where this lives in code

| quantity | code |
|---|---|
| $s, a$, angle, rot-score of a full block | `elm.rotation_decompose` (`ds`, `anti`, `angle_deg`, `rot_score`) |
| per-window $s, a$, angle | `elm.rotation_decompose` on a stacked `(n, 2, 2)` array |
| iso/gen/shear complex split | derivation only; no longer implemented in the package |

`rotation_decompose` defines `rot_score = anti**2 / (anti**2 + sym_od**2 + diag_d**2 + 1e-12)`,
which is exactly the boxed formula with `sym_od =` $\gamma_{\rm o}$, `diag_d =` $\gamma_{\rm d}$.

**Numbers to expect (noiseless bs=2).** The averaged blocks are near-pure scaled
rotations: $J_{\rm recent}\approx 1.85\,R(\sim5^\circ)$, $J_{\rm prev}\approx
-0.9\,R(\cdots)$, both with rot-score $\approx 1$ (measured with `rotation_decompose`; only the
noisy counterpart is plotted, as `fig-jacobian-rotation`).

---

## B. The Verlet-inertia residual

The physical reference is the second-order **Störmer–Verlet** integrator of
inverse-square gravity ($GM=1$):

$$
\hat{\mathbf r}_{t+1}
= \underbrace{2\,\mathbf r_t - \mathbf r_{t-1}}_{\text{inertial (constant-velocity)}}
\;\underbrace{-\;\Delta t^2\,\dfrac{\mathbf r_t}{|\mathbf r_t|^3}}_{\text{gravitational kick}}.
$$

As a per-token linear map the inertial part is exactly

$$
J_{\rm recent}^{\rm Verlet} = 2I,\qquad J_{\rm prev}^{\rm Verlet} = -I,
$$

which sits at the **Inertial point** $(c_{\rm prev}, c_{\rm recent}) = (-1, 2)$ in
isotropic-coefficient space. The noiseless model lands there
($\approx(-0.9, +1.85)$; fig `fig-verlet-noiseless`), unlike the noisy model
which sits on the persistence line $c_{\rm prev}+c_{\rm recent}=1$ at
$(0.36, 0.62)$.

### The residual blocks

Everything Newtonian *beyond* dead-reckoning lives in the deviation from the
ideal inertial weights:

$$
\boxed{\;
\mathrm{d}J_{\rm recent} = J_{\rm recent} - 2I,\qquad
\mathrm{d}J_{\rm prev}   = J_{\rm prev}   + I.
\;}
$$

(Code: `dJr = all_blocks[:, T-1] - 2*I2`, `dJp = all_blocks[:, T-2] + I2`, as
`elm.residual_structure` computes them for `fig-residual-structure`. The same
subtraction, taken on the *products* rather than the operators, is what
`elm.kick_terms` reports for `fig-kick-terms`.)

Acting on the two input positions, the residual is the model's **non-inertial
correction**:

$$
\mathrm{d}J_{\rm recent}\,\mathbf r_t + \mathrm{d}J_{\rm prev}\,\mathbf r_{t-1}
\;=\; \hat{\mathbf r}_{t+1} - (2\mathbf r_t - \mathbf r_{t-1})
\;\stackrel{?}{\approx}\; -\Delta t^2\,\frac{\mathbf r_t}{|\mathbf r_t|^3}.
$$

### Behavioural check (attribution-free — do this first)

Before decomposing the residual we confirm, with **no reference to the
Jacobian**, that the model's non-inertial output *is* Newtonian gravity. From
the second difference of the model's own prediction,

$$
\mathbf a_{\rm eff} = \frac{\hat{\mathbf r}_{t+1} - 2\mathbf r_t + \mathbf r_{t-1}}{\Delta t^2},
$$

a genuine force integrator gives $\mathbf a_{\rm eff} = -\mathbf r_t/|\mathbf r_t|^3$.
Measured (fig `fig-newton-behavioral`, cell code):

- **Direction:** $\cos\angle(\mathbf a_{\rm eff}, -\hat{\mathbf r}_t) \approx 0.999$ — points at the sun.
- **Magnitude:** $|\mathbf a_{\rm eff}| \sim r^{p}$ with $p \approx -1.9$ (Newtonian $-2$), intercept $GM_{\rm est}\approx 1.04$.
- **Components:** $\operatorname{corr}(\mathbf a_{\rm eff}, -\mathbf r_t/|\mathbf r_t|^3) \approx 0.98$ per axis.

This is the ground truth the §B decomposition must be consistent with: the
residual has to reconstruct exactly this kick.

---

## C. Common / difference (bar / delta) mode decomposition of the residual

We now ask *how the residual is organised across the two tokens*. §A gives us,
for **each** residual block, its invariant coefficients — the isotropic scale
$s$ and rotation generator $a$ of $\mathrm dJ$ (same formulas as §A, now applied
to $\mathrm dJ$ instead of $J$). That leaves a *pair* of numbers per coefficient:
one from the recent block, one from the previous block, e.g.
$(s_{\rm recent}, s_{\rm prev})$ and $(a_{\rm recent}, a_{\rm prev})$.

**Why a common/difference basis.** A window supplies two positions,
$\mathbf r_t$ (recent) and $\mathbf r_{t-1}$ (previous). Any operator built from
per-token coefficients acts on *both*, and there are two physically distinct
things such a pair can encode:

- their **sum** $\mathbf r_t + \mathbf r_{t-1}$ — a position-like quantity
  (twice the midpoint), and
- their **difference** $\mathbf v \equiv \mathbf r_t - \mathbf r_{t-1}$ — the
  one-step velocity.

The natural way to expose which of the two an operator uses is to rotate the
raw $(\text{recent}, \text{prev})$ coefficient pair into its symmetric and
antisymmetric parts — the **common** and **difference** modes. This is just a
change of basis on $\mathbb R^2$ (a $45^\circ$ rotation of the coefficient pair),
so it is lossless: no information is added or discarded, only reorganised so the
"acts-on-position" and "acts-on-velocity" content sit in separate coordinates.
Concretely, for any per-token coefficient $c$ (we use $c = s$ and $c = a$):

$$
\boxed{\;
\bar c = \tfrac12\big(c_{\rm recent} + c_{\rm prev}\big) \quad\text{(common mode)},\qquad
\Delta c = \tfrac12\big(c_{\rm recent} - c_{\rm prev}\big) \quad\text{(difference mode)}.
\;}
$$

so that $c_{\rm recent} = \bar c + \Delta c$ and $c_{\rm prev} = \bar c - \Delta c$.
The common mode is the shared, symmetric part of the two tokens' action; the
difference mode is the antisymmetric part. As the identity below makes precise,
$\bar c$ multiplies the *position* sum and $\Delta c$ multiplies the *velocity* —
so measuring which mode dominates tells us, model-agnostically, whether the
residual operator reads positions or velocity.

> **Notation note.** The code labels the residual's isotropic coefficient `d`
> (for *deviation*); here we call it $s$ throughout, matching §A. So the code's
> $\bar d, \Delta d$ are this doc's $\bar s, \Delta s$.

(Code, `elm.residual_structure`: `abar, adiff = median((a_r+a_p)/2), median((a_r-a_p)/2)`
and likewise `dbar, ddiff` for the isotropic mode. The residual coefficients come
from its `parts(M)` helper: `d = (dJ00+dJ11)/2` $= s$,
`a = (dJ10-dJ01)/2`, `od = (dJ01+dJ10)/2` $= \gamma_{\rm o}$,
`dd = (dJ00-dJ11)/2` $= \gamma_{\rm d}$, `score = a**2/(a**2+od**2+dd**2)`.)

### The key identity: common mode acts on position, difference mode on velocity

Take the residual's isotropic contribution and substitute the split:

$$
\begin{aligned}
s_{\rm recent}\,\mathbf r_t + s_{\rm prev}\,\mathbf r_{t-1}
&= (\bar s + \Delta s)\,\mathbf r_t + (\bar s - \Delta s)\,\mathbf r_{t-1}\\
&= \bar s\,(\mathbf r_t + \mathbf r_{t-1}) \;+\; \Delta s\,(\mathbf r_t - \mathbf r_{t-1})\\
&= \underbrace{\bar s\,(\mathbf r_t + \mathbf r_{t-1})}_{\text{common} \to \text{position/midpoint}}
 \;+\; \underbrace{\Delta s\,\mathbf v}_{\text{difference} \to \text{velocity}},
\qquad \mathbf v \equiv \mathbf r_t - \mathbf r_{t-1}.
\end{aligned}
$$

Identically for the rotation part (factor $R$ out first):

$$
a_{\rm recent}\,R\,\mathbf r_t + a_{\rm prev}\,R\,\mathbf r_{t-1}
= R\big[\,\bar a\,(\mathbf r_t + \mathbf r_{t-1}) + \Delta a\,\mathbf v\,\big].
$$

So a **vanishing common mode** ($\bar c \approx 0$) is exactly the statement
that the operator acts on the *velocity* $\mathbf v$, not on either position
alone. That is the structural claim of this section.

### Measured (noiseless bs=2)

Both residual blocks are near-pure scaled rotations (rot-score $\approx 1$,
shear negligible — `fig-residual-structure`b), and the coefficients are
**equal-and-opposite** across the token pair:

- rotation generator: $\bar a \approx 0$, with $|\bar a|/(|\bar a|+|\Delta a|)\approx 0.02$;
- isotropic scale: $\bar s \approx 0$, with $|\bar s|/(|\bar s|+|\Delta s|)\approx 0.06$.

Both difference amplitudes grow toward small $r$ (the $1/r^2$ strengthening a
force needs). With $\bar a\approx0,\ \bar s\approx0$ the residual collapses to
**two velocity operators**:

$$
\text{rotation antisymmetry} = \Delta a\,R\,\mathbf v,
\qquad
\text{isotropic antisymmetry} = \Delta s\,\mathbf v.
$$

### Direction test — which antisymmetry is the gravitational pull?

Each mode is now a single vector; resolve it onto the radial (inward-gravity)
axis $-\hat{\mathbf r}_t$ and the tangential axis (`rot_vec = Δa · (v @ Rᵀ)`,
`iso_vec = Δs · v`; derivation only — this test is not plotted in the pub):

- **Rotation antisymmetry** $\Delta a\,R\,\mathbf v$ — velocity turned $90^\circ$.
  $\cos(R\mathbf v, -\hat{\mathbf r}_t)\approx 0.98$: turning a (near-)tangential
  orbital velocity by $90^\circ$ lands on the radial direction, so this mode
  carries the **sun-directed** pull.
- **Isotropic antisymmetry** $\Delta s\,\mathbf v$ — a scaling of $\mathbf v$,
  hence *along* $\mathbf v$ (tangential). $\cos \approx 0.09$: it strengthens a
  tangential term, **not** the gravitational pull.

Resolved on the radial/tangential plane the two clusters are near-orthogonal
the rotation antisymmetry is specifically the
one supplying gravity. The single-mode alignment ($\cos\approx0.98$, i.e.
$\sim13^\circ$) is short of the full correction's $\cos\approx0.999$; the gap is
closed by the remaining terms, whose full channel-level accounting is left to
future work.

---

## D. End-to-end workflow (noiseless model), in order

1. Load the noiseless checkpoint (`noise_scale = 0.0`), `float64`, eval mode.
2. `NT.collect_terms(model, block_size=2, n_windows=N, seed, offset)` → `all_blocks (n,2,2,2)`, `coeffs = tr/D`, plus the content/positional/radius terms.
3. **Isotropic coefficients** (§A, $s$): scatter $(c_{\rm prev}, c_{\rm recent})$ → lands on Inertial point $(-1,2)$. (`fig-verlet-noiseless`)
4. **§A on full blocks:** `rotation_decompose` → averaged blocks are scaled rotations, rot-score ≈ 1. (not plotted for the noiseless model; cf. `fig-jacobian-rotation`)
5. **Behavioural gravity test** (§B, attribution-free): $\mathbf a_{\rm eff}$ points sunward, $\propto 1/r^2$, matches $-\mathbf r/|\mathbf r|^3$. (`fig-newton-behavioral`)
6. **Residual blocks** (§B): $\mathrm dJ_{\rm recent}=J_{\rm recent}-2I$, $\mathrm dJ_{\rm prev}=J_{\rm prev}+I$.
7. **§A on residual + §C:** per-block rot-score ≈ 1; common/difference split gives $\bar a\approx\bar s\approx0$ (equal-and-opposite). (`fig-residual-structure`)
8. **Direction of the two antisymmetric velocity operators** (§C): rotation → radial/sunward ($\cos\approx0.98$); isotropic → tangential ($\cos\approx0.09$). (derivation only, not plotted)

---

## E. Points worth an extra look (for the reviewer)

- **Sign / index conventions.** Confirm $J_{ij}=\partial\text{out}_i/\partial\text{in}_j$ and
  index `0 = x`, `1 = y` are consistent with the generator sign
  $a=\tfrac12(J_{10}-J_{01})$ and $R=[[0,-1],[1,0]]$ (a $+90^\circ$ CCW turn).
- **rot-score excludes the isotropic scale $s$.** It is a ratio of
  rotation vs *shear* energy only; a large isotropic scale does not inflate it.
  Check this is the intended question ("turn vs distortion", not "turn vs size").
- **Median vs mean.** The common/difference summaries use the **median** over
  windows (robust to the aphelion tail where the $1/\Delta t^2$ second difference
  amplifies noise). Worth confirming the mean tells the same story.
- **Antisymmetry is a per-window claim.** $\bar a, \bar s$ vanish in $\sim100\%$
  of windows, not merely on average — the cancellation is not a pooling artifact
  (contrast the *averaged* full blocks in step 4, where averaging alone could
  manufacture a rotation).
- **Closure.** Every decomposition is exact by construction (iso + gen + shear +
  positional reconstructs the model output to $\sim10^{-9}$); the physics claims
  are about how that exact sum is *organised*, not about approximation error.
