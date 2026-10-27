"""Analysis code for the Newton-vs-Kepler learned world model publication.

Modules
-------
data_generation  Generate the Kepler trajectory dataset (CLI: python -m analysis.data_generation)
data             Load trajectories and orbital parameters; chop and sample windows
model            The GPT-2-style GPTCV, plus the detached-nonlinearity ELM machinery
training         Train the checkpoints the analyses use (CLI: python -m analysis.training)
probing          Linear probes for the Newtonian and Keplerian target variables
elm              Equivalent Linear Mapping analysis and the publication's figure helpers

`index.ipynb` is the publication and the entry point for the analysis.
"""
