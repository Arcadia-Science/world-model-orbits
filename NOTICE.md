# Notice

The code in this repository is released under the MIT License (see `LICENSE`),
with the exception of the portions described below.

## Third-party code

Parts of the `analysis` package (`src/analysis/`) are adapted from:

> Liu et al., *"From Kepler to Newton: Inductive Biases Guide Learned World
> Models in Transformers"*, [arXiv:2602.06923](https://arxiv.org/abs/2602.06923).
> Source: <https://github.com/KindXiaoming/newton-kepler>
> Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Changes were made. Each affected file carries the same attribution in its header;
they are collected here:

| File | Relationship to the upstream work |
|---|---|
| `data_generation.py` | Verbatim copy of `generate_kepler_cv.py`, repackaged into the `analysis` package. |
| `data.py` | Trajectory loading and windowing, extracted from the upstream training script. |
| `probing.py` | Linear and geometry probing, extracted from the upstream training script. |
| `training.py` | Training and rollout code from the upstream training script; `main()` rewritten to train the three models this publication analyses. |
| `model.py` | The upstream model, extended with Equivalent Linear Mapping (ELM) Jacobian detachment and `bias=False`. |

All other files in this repository, including the publication notebook
(`index.ipynb`) and the analysis in `src/analysis/elm.py`, are original work by
Arcadia Science under the MIT License.
