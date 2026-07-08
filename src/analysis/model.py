"""
Equivalent Linear Mapping (ELM) extension of model_cv.py.

Adds a `detach_for_elm` flag to GPTCV that makes the forward pass
"pointwise linear" at a given input x* by detaching the input-dependent
scaling terms from the gradient computation.  The detached Jacobian J+
then satisfies:

    y* = sum_i  J+_i(x*) @ x*_i          (exact reconstruction)

Three nonlinear components require detachment:

  1. LayerNorm   – detach the RMS denominator:
                   norm(x) = x / rms(x*)
                   gradient sees only the numerator x as linear.

  2. SiLU        – detach the sigmoid gate:
                   SiLU(x) = x * sigmoid(beta * x*)
                   gradient sees only the linear x factor.

  3. Softmax attn – detach the attention weight matrix A = softmax(QK^T/sqrt(d)):
                   output = A(x*) @ V(x)
                   gradient flows only through V = W_V x.

All other operations (Linear layers, residual adds, positional embedding,
input/output projections) are already linear and need no modification.

Usage
-----
    # standard forward (unchanged behaviour)
    model = GPTCV(config)
    preds, loss = model(x, targets)

    # ELM forward – pointwise linear at x*
    model.detach_for_elm = True
    model.eval()
    with torch.enable_grad():
        preds_elm = model(x_star)[0]   # identical numerically to standard forward
    J = torch.autograd.functional.jacobian(lambda inp: model(inp)[0], x_star)

    # verify reconstruction
    model.detach_for_elm = False

Notes
-----
- `detach_for_elm` must be False during training (default).
- The model must have bias=False throughout (GPTConfigCV.bias = False) for
  the homogeneity argument to hold.  This is already the case in the
  no-bias ablation used by legacy/kepler_cv_blocksize_no_bias.py.
- Reconstruction error should be < ~1e-10 at float64 precision.
"""

# ---------------------------------------------------------------------------
# Adapted from Liu et al., "From Kepler to Newton: Inductive Biases Guide
# Learned World Models in Transformers" (arXiv:2602.06923), CC BY 4.0
# (https://creativecommons.org/licenses/by/4.0/).
# Source: https://github.com/KindXiaoming/newton-kepler
# Modification: extends the upstream model with Equivalent Linear Mapping (ELM)
# detachment and bias=False. Full attribution in NOTICE.md.
# ---------------------------------------------------------------------------

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Helper: detached LayerNorm
# ---------------------------------------------------------------------------

def layernorm_detached(x, weight, eps=1e-5):
    """
    LayerNorm that is pointwise-linear at the current input x.

    The RMS normalisation constant is computed from x but detached from the
    computational graph, so the gradient with respect to x only flows through
    the numerator, making the operation linear at this operating point.

    norm(x) = x / rms(x).detach()  *  weight

    Returns the same numerical value as standard LayerNorm (bias=False).
    """
    # variance over last dim (same as F.layer_norm with bias=None)
    # We use RMS-style normalisation matching F.layer_norm's behaviour
    # F.layer_norm: (x - mean) / sqrt(var + eps) * weight
    # For bias=False models trained with F.layer_norm we match exactly.
    mean = x.mean(dim=-1, keepdim=True)
    var  = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    # Only the variance-based denominator is nonlinear and must be frozen.
    # Mean-centering (x - mean) is a linear, degree-1 homogeneous operation
    # (the centering matrix C = I - 11^T/d), so it must stay in the gradient
    # path. Detaching the mean would drop C from the Jacobian and inject a
    # spurious constant -mean*weight/denom into the reconstruction residual.
    denom = (var + eps).sqrt().detach()
    x_norm = (x - mean) / denom
    return x_norm * weight


# ---------------------------------------------------------------------------
# Helper: detached SiLU
# ---------------------------------------------------------------------------

def silu_detached(x):
    """
    SiLU that is pointwise-linear at the current input x.

    SiLU(x) = x * sigmoid(x)

    We detach sigmoid(x) so the gradient only flows through the linear x:
        d/dx [x * sigmoid(x*).detach()] = sigmoid(x*)

    Returns the same numerical value as nn.SiLU()(x).
    """
    gate = torch.sigmoid(x).detach()   # scalar gate frozen at x*
    return x * gate


# ---------------------------------------------------------------------------
# Model modules (drop-in replacements when detach_for_elm=True)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm with optional bias and optional ELM detachment."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None
        # set by GPTCV; controls whether detached forward is used
        self.detach_for_elm = False

    def forward(self, x):
        if self.detach_for_elm:
            out = layernorm_detached(x, self.weight)
            if self.bias is not None:
                out = out + self.bias   # bias is a constant, does not break linearity
            return out
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd,     bias=config.bias)
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.dropout = config.dropout
        self.alpha   = config.attention_alpha
        self.flash   = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                  .view(1, 1, config.block_size, config.block_size)
        )
        # set by GPTCV
        self.detach_for_elm = False

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.detach_for_elm:
            # --- ELM attention: freeze A(x*) entirely, keep V linear in x ---
            # Q and K both depend on x, so detach them before computing scores.
            # If we only detached softmax(QK^T) the QK^T product would still
            # carry gradients w.r.t. x through q and k, breaking linearity.
            q_d = q.detach()
            k_d = k.detach()
            att = (q_d @ k_d.transpose(-2, -1)) * (1.0 / math.sqrt(k_d.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

            if self.alpha != 1.0:
                positions      = torch.arange(1, T + 1, dtype=att.dtype, device=att.device)
                position_scale = torch.log(positions) ** self.alpha
                att = att * position_scale.view(1, 1, T, 1)

            # att is fully detached from x; softmax is just a monotone transform
            att_weights = F.softmax(att, dim=-1)   # frozen A(x*), no grad w.r.t. x
            # V = W_V x  is still linear in x (gradient flows here)
            y = att_weights @ v     # A(x*) @ V(x) -- linear in x

        elif self.flash and self.alpha == 1.0:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            if self.alpha != 1.0:
                positions      = torch.arange(1, T + 1, dtype=att.dtype, device=att.device)
                position_scale = torch.log(positions) ** self.alpha
                att = att * position_scale.view(1, 1, T, 1)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y   = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.silu   = nn.SiLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        # set by GPTCV
        self.detach_for_elm = False

    def forward(self, x):
        x = self.c_fc(x)
        if self.detach_for_elm:
            x = silu_detached(x)
        else:
            x = self.silu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfigCV:
    block_size:      int   = 1024
    n_layer:         int   = 12
    n_head:          int   = 12
    n_embd:          int   = 768
    dropout:         float = 0.0
    bias:            bool  = True
    input_dim:       int   = 1
    attention_alpha: float = 0.0


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class GPTCV(nn.Module):
    """
    GPT model for continuous variables with optional ELM detachment.

    Set `model.detach_for_elm = True` before computing the Jacobian.
    This propagates the flag to every LayerNorm, CausalSelfAttention,
    and MLP submodule so their forwards become pointwise-linear.
    """

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config
        self._detach_for_elm = False   # internal flag

        self.transformer = nn.ModuleDict(dict(
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.input_embedding = nn.Linear(config.input_dim, config.n_embd, bias=config.bias)
        self.output_head     = nn.Linear(config.n_embd,    config.input_dim, bias=config.bias)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.6fM" % (self.get_num_params() / 1e6,))

    # ------------------------------------------------------------------
    # ELM flag property – propagates to all submodules on assignment
    # ------------------------------------------------------------------

    @property
    def detach_for_elm(self):
        return self._detach_for_elm

    @detach_for_elm.setter
    def detach_for_elm(self, value: bool):
        self._detach_for_elm = value
        # propagate to every submodule that has the flag
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, 'detach_for_elm'):
                module.detach_for_elm = value

    # ------------------------------------------------------------------
    # Standard helpers (unchanged from original model_cv.py)
    # ------------------------------------------------------------------

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed_continuous(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        b, t, d = x.size()
        assert d == self.config.input_dim
        return self.input_embedding(x)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, targets=None):
        device = x.device

        if x.dim() == 2:
            x = x.unsqueeze(-1)

        b, t, d = x.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        assert d == self.config.input_dim

        pos     = torch.arange(0, t, dtype=torch.long, device=device)
        x_emb   = self.embed_continuous(x)
        pos_emb = self.transformer.wpe(pos)
        x       = self.transformer.drop(x_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            predictions = self.output_head(x)
            if targets.dim() == 2:
                targets = targets.unsqueeze(-1)
            if targets.size(1) < predictions.size(1):
                predictions = predictions[:, -targets.size(1):, :]
            elif targets.size(1) > predictions.size(1):
                raise ValueError(
                    f"Target sequence length {targets.size(1)} > prediction length {predictions.size(1)}"
                )
            predictions_flat = predictions.reshape(-1)
            targets_flat     = targets.reshape(-1)
            loss = F.mse_loss(predictions_flat, targets_flat)
        else:
            predictions = self.output_head(x)
            loss = None

        return predictions, loss

    def forward_from_embedding(self, h):
        """
        Run the transformer starting from the post-sum embedding h.

        h is the already-summed content + positional embedding, i.e. exactly
        the tensor forward() feeds into transformer.drop:

            h_t = W_in @ x_t + wpe[t]          (= x_emb + pos_emb)

        This is the natural differentiation point for ELM on a model with
        *additive* positional embeddings: the map h -> output is genuinely
        degree-1 homogeneous after detachment (no additive constant), so the
        detached Jacobians reconstruct the output exactly, whereas
        differentiating w.r.t. the raw input x leaves the wpe constant in an
        un-homogeneous affine offset.

        Honors detach_for_elm automatically (the flag lives on the submodules
        this method calls). No targets / loss branch -- analysis only.

        Args:
            h: tensor of shape (B, T, n_embd). Use B=1 for Jacobian analysis.

        Returns:
            predictions: tensor of shape (B, T, input_dim)
        """
        x = self.transformer.drop(h)          # identity in eval
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.output_head(x)

    # ------------------------------------------------------------------
    # ELM utilities
    # ------------------------------------------------------------------

    def compute_positional_contribution(self, x_star):
        """
        Compute the output contribution from positional embeddings alone.

        This runs a forward pass with the same sequence length as x_star,
        but with zero content (x=0), isolating the positional embedding effect.

        Args:
            x_star: input tensor of shape (1, T, input_dim) or (1, T)
                    (only used to determine sequence length T)

        Returns:
            pos_contribution: flattened output from positional embeddings only,
                             shape (T * input_dim,)
        """
        if x_star.dim() == 2:
            x_star = x_star.unsqueeze(-1)

        T = x_star.shape[1]
        input_dim = x_star.shape[2]
        device = x_star.device

        # Create zero content input (same shape as x_star)
        x_zero = torch.zeros(1, T, input_dim, dtype=x_star.dtype, device=device)

        # Forward pass with zero content (only positional embeddings contribute)
        with torch.no_grad():
            y_pos_only, _ = self.forward(x_zero)

        return y_pos_only.reshape(-1)

    def compute_detached_jacobians(self, x_star):
        """
        Compute the detached Jacobian matrices J+_i for each input token position.

        For a sequence x* of shape (1, T, input_dim), returns a list of T matrices
        each of shape (output_dim_flat, input_dim), where output_dim_flat is the
        total number of output scalars (1 * T * input_dim for full-sequence output).

        Args:
            x_star: input tensor of shape (1, T, input_dim) or (1, T), dtype float64
                    recommended for numerical precision.

        Returns:
            jacobians:         list of T tensors, each of shape (T * input_dim, input_dim)
                              J+_i such that sum_i J+_i @ x*_i + pos + residual == y*
            y_star:           model output tensor, shape (1, T, input_dim)
            recon_err:        relative reconstruction error (scalar) for verification
            pos_contribution: flattened output from positional embeddings only
            b_residual:       residual affine offset (should be ~0 if Jacobian is correct)
        """
        assert x_star.shape[0] == 1, "Batch size must be 1 for Jacobian computation"

        if x_star.dim() == 2:
            x_star = x_star.unsqueeze(-1)   # (1, T, 1)

        T         = x_star.shape[1]
        input_dim = x_star.shape[2]


        self.eval()
        pos_contribution = self.compute_positional_contribution(x_star)

        # Enable ELM mode
        self.detach_for_elm = True
        self.eval()

        # Forward pass to get y* (numerical output is identical to standard forward)
        with torch.no_grad():
            y_star, _ = self.forward(x_star)
        y_star_flat = y_star.reshape(-1)   # (T * input_dim,)

        # Compute Jacobian w.r.t. each token position separately
        # x_star shape (1, T, input_dim) – we differentiate w.r.t. each (1, 1, input_dim) slice
        jacobians = []

        # Build each per-token Jacobian by constructing a fresh closure that
        # captures `idx` by value (default-argument trick), not by reference.
        # The closure also builds the full input tensor from scratch rather than
        # cloning-then-assigning, so the autograd graph is clean.
        def make_forward_flat(idx, x_star_fixed):
            """Return a function that replaces position idx and runs forward."""
            # rows that are NOT idx are constants; only position idx is a variable
            before = x_star_fixed[0, :idx, :].detach()          # (idx, input_dim)
            after  = x_star_fixed[0, idx+1:, :].detach()        # (T-idx-1, input_dim)
            def forward_flat(x_i):
                # x_i: (input_dim,)  -- the live variable for autograd
                x_in = torch.cat([
                    before,
                    x_i.unsqueeze(0),
                    after,
                ], dim=0).unsqueeze(0)   # (1, T, input_dim)
                with torch.enable_grad():
                    out, _ = self.forward(x_in)
                return out.reshape(-1)   # (T * input_dim,)
            return forward_flat

        for i in range(T):
            x_i = x_star[0, i, :].detach().requires_grad_(True)   # (input_dim,)
            J_i = torch.autograd.functional.jacobian(
                make_forward_flat(i, x_star), x_i,
                create_graph=False, strict=False
            )
            jacobians.append(J_i)   # (T * input_dim, input_dim)

        # Affine offset correction - separate positional from residual
        # Total affine offset: b(x*) = y* - J+(x*) @ x*
        J_x_star = sum(
            jacobians[i] @ x_star[0, i, :]
            for i in range(T)
        )

        # Test reconstruction with Jacobian + positional (without residual)
        recon = J_x_star + pos_contribution
        recon_err = (recon - y_star_flat).norm() / (y_star_flat.norm() + 1e-30)

        # Compute residual - if small, Jacobian is working correctly
        b_residual = y_star_flat - (J_x_star + pos_contribution)

        recon_err = recon_err.item()

        # Restore standard mode
        self.detach_for_elm = False

        return jacobians, y_star, recon_err, pos_contribution, b_residual

    def compute_detached_jacobians_embedding(self, x_star):
        """
        Embedding-level ELM decomposition (differentiates downstream of the
        positional-embedding sum, unlike compute_detached_jacobians which
        differentiates w.r.t. the raw input x).

        With h_t = W_in @ x_t + wpe[t], the map h -> output is degree-1
        homogeneous after ELM detachment, so reconstruction is EXACT:

            y* = Sum_t J_{h,t} @ h_t                       (recon_err ~ 1e-16)

        and splits cleanly through the SAME x*-frozen Jacobian:

            y* = Sum_t (J_{h,t} W_in) @ x_t  +  Sum_t J_{h,t} @ wpe[t]
                 \-------- content --------/     \------ positional ------/

        Because the input embedding is linear, J_{x,t} = J_{h,t} @ W_in
        recovers the per-(x,y)-coordinate contribution with no residual.

        Args:
            x_star: input tensor of shape (1, T, input_dim) or (1, T), float64
                    recommended (must match the model dtype).

        Returns:
            jacobians_h:             list of T tensors, each (T*input_dim, n_embd)  = J_{h,t}
            jacobians_x:             list of T tensors, each (T*input_dim, input_dim) = J_{h,t} @ W_in
            y_star:                  model output tensor, shape (1, T, input_dim)
            recon_err:               relative error of Sum_t J_{h,t} @ h_t vs y* (~1e-16)
            content_contribution:    (T*input_dim,)  Sum_t (J_{h,t} W_in) @ x_t
            positional_contribution: (T*input_dim,)  Sum_t J_{h,t} @ wpe[t]
            content_norms:           list of T floats, ||(J_{h,t} W_in) @ x_t||
            positional_norms:        list of T floats, ||J_{h,t} @ wpe[t]||
        """
        assert x_star.shape[0] == 1, "Batch size must be 1 for Jacobian computation"

        if x_star.dim() == 2:
            x_star = x_star.unsqueeze(-1)   # (1, T, 1)

        T         = x_star.shape[1]
        input_dim = x_star.shape[2]
        device    = x_star.device
        assert T <= self.config.block_size, \
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # Save the detach flag so we can restore the caller's state exactly.
        prev_flag = self.detach_for_elm
        self.eval()

        try:
            # Build h* = x_emb + pos_emb -- the post-sum embedding entering the
            # transformer blocks (mirrors forward(): embed_continuous + wpe).
            pos     = torch.arange(0, T, dtype=torch.long, device=device)
            x_emb   = self.embed_continuous(x_star)          # (1, T, n_embd)
            pos_emb = self.transformer.wpe(pos)              # (T, n_embd)
            h_star  = (x_emb + pos_emb).detach()             # (1, T, n_embd)

            # Enable ELM mode and get y* (numerically identical to forward(x*)).
            self.detach_for_elm = True
            self.eval()
            with torch.no_grad():
                y_star = self.forward_from_embedding(h_star)  # (1, T, input_dim)
            y_star_flat = y_star.reshape(-1)                  # (T * input_dim,)

            # Per-position Jacobian w.r.t. h_t. Same closure pattern as the raw
            # version's make_forward_flat, but the live variable is the n_embd
            # embedding vector and we go through forward_from_embedding.
            def make_forward_flat_h(idx, h_fixed):
                before = h_fixed[0, :idx, :].detach()         # (idx, n_embd)
                after  = h_fixed[0, idx+1:, :].detach()       # (T-idx-1, n_embd)
                def forward_flat(h_i):
                    h_in = torch.cat([
                        before,
                        h_i.unsqueeze(0),
                        after,
                    ], dim=0).unsqueeze(0)                    # (1, T, n_embd)
                    with torch.enable_grad():
                        out = self.forward_from_embedding(h_in)
                    return out.reshape(-1)                    # (T * input_dim,)
                return forward_flat

            jacobians_h = []
            for i in range(T):
                h_i = h_star[0, i, :].detach().requires_grad_(True)   # (n_embd,)
                J_i = torch.autograd.functional.jacobian(
                    make_forward_flat_h(i, h_star), h_i,
                    create_graph=False, strict=False
                )
                jacobians_h.append(J_i)   # (T * input_dim, n_embd)

            # J_{x,t} = J_{h,t} @ W_in  (chain rule through the linear embedding)
            W_in = self.input_embedding.weight                # (n_embd, input_dim)
            jacobians_x = [J_h @ W_in for J_h in jacobians_h]

            # Exact content / positional split through the same frozen J_h.
            content_contribution = sum(
                jacobians_x[i] @ x_star[0, i, :] for i in range(T)
            )                                                 # (T * input_dim,)
            positional_contribution = sum(
                jacobians_h[i] @ pos_emb[i] for i in range(T)
            )                                                 # (T * input_dim,)
            content_norms = [
                (jacobians_x[i] @ x_star[0, i, :]).norm().item() for i in range(T)
            ]
            positional_norms = [
                (jacobians_h[i] @ pos_emb[i]).norm().item() for i in range(T)
            ]

            # Reconstruction: content + positional == Sum_t J_{h,t} @ h_t.
            recon = content_contribution + positional_contribution
            recon_err = (
                (recon - y_star_flat).norm() / (y_star_flat.norm() + 1e-30)
            ).item()
        finally:
            # Restore the caller's detach mode (do not clobber it).
            self.detach_for_elm = prev_flag

        return (jacobians_h, jacobians_x, y_star, recon_err,
                content_contribution, positional_contribution,
                content_norms, positional_norms)

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, x, max_new_tokens):
        input_was_2d = x.dim() == 2
        if input_was_2d:
            x = x.unsqueeze(-1)

        for _ in range(max_new_tokens):
            x_cond      = x if x.size(1) <= self.config.block_size else x[:, -self.config.block_size:]
            predictions, _ = self(x_cond)
            x_next      = predictions[:, -1, :].unsqueeze(1)
            x           = torch.cat((x, x_next), dim=1)

        if input_was_2d:
            x = x.squeeze(-1)
        return x

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path, optimizer=None, step=None, extra=None):
        """
        Save model weights and config to a .pt checkpoint.

        Args:
            path:      file path to write (e.g. 'checkpoints/model_step1000.pt')
            optimizer: optional optimizer – its state_dict is saved alongside
            step:      optional int, training step at time of save
            extra:     optional dict of anything else to store in the checkpoint

        The checkpoint dict has the following keys:
            'model_state_dict'   – model weights
            'config'             – GPTConfigCV dataclass (as a plain dict)
            'step'               – training step (or None)
            'optimizer_state_dict' – optimizer state (or None)
            + any keys from `extra`
        """
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        config_dict = {
            'block_size':      self.config.block_size,
            'n_layer':         self.config.n_layer,
            'n_head':          self.config.n_head,
            'n_embd':          self.config.n_embd,
            'dropout':         self.config.dropout,
            'bias':            self.config.bias,
            'input_dim':       self.config.input_dim,
            'attention_alpha': self.config.attention_alpha,
        }
        checkpoint = {
            'model_state_dict':     self.state_dict(),
            'config':               config_dict,
            'step':                 step,
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        }
        if extra is not None:
            checkpoint.update(extra)

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}"
              + (f" (step {step})" if step is not None else ""))

    @classmethod
    def load_checkpoint(cls, path, device=None, strict=True):
        """
        Load a GPTCV model from a checkpoint saved by save_checkpoint().

        Args:
            path:   path to the .pt checkpoint file
            device: torch device to map tensors onto (default: cpu)
            strict: passed to load_state_dict

        Returns:
            model:      GPTCV instance with weights loaded
            checkpoint: the full checkpoint dict (contains 'step',
                        'optimizer_state_dict', etc.)

        Example
        -------
            model, ckpt = GPTCV.load_checkpoint('checkpoints/model_step1000.pt',
                                                  device='cuda')
            # resume optimizer
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        """
        if device is None:
            device = torch.device('cpu')

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Reconstruct config from the stored dict
        config_dict = checkpoint['config']
        config = GPTConfigCV(**config_dict)

        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        model.to(device)
        model.eval()

        step = checkpoint.get('step', None)
        print(f"Checkpoint loaded from {path}"
              + (f" (step {step})" if step is not None else ""))
        return model, checkpoint


# ---------------------------------------------------------------------------
# Standalone convenience function
# ---------------------------------------------------------------------------

def load_from_checkpoint(path, device=None, strict=True):
    """
    Load a GPTCV model from a checkpoint.  Thin wrapper around
    GPTCV.load_checkpoint() for cases where you don't want to import
    the class name explicitly.

    Args:
        path:   path to the .pt checkpoint file
        device: torch device string or object (default: 'cpu')
        strict: passed to load_state_dict

    Returns:
        model:      GPTCV instance, eval mode, on requested device
        checkpoint: full checkpoint dict

    Example
    -------
        from model_cv_elm import load_from_checkpoint
        model, ckpt = load_from_checkpoint('run_block2/best_model.pt', device='cuda')
        model.verify_elm_reconstruction(x_star.double())
    """
    return GPTCV.load_checkpoint(path, device=device, strict=strict)