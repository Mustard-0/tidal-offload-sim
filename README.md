# Tidal offload simulation

CPU experiments for a cross-slot satellite-terrestrial edge simulation with
8 base stations, 3 satellites, heterogeneous tasks and class-specific shuffle
pools. Workloads and validation streams are paired across algorithms.

The runtime and finite experiment budget are frozen by
`experiments/hybrid_campaign.py`. The coordinator assigns disjoint cells to
local processes and standard GitHub-hosted Linux jobs. Stage tags explicitly
start a validation screen; formal training requires the saved screen evidence
to pass a reproducible raw-data gate.

Python 3.9.25; CPU PyTorch 2.5.1+cpu. Other dependencies are in requirements.txt.
Model checkpoints, raw simulated task CSVs and logs are stored in Release
assets. The workflow does not use paid larger runners or Actions artifact
storage. Ordinary branch pushes do not start training.

TMPPO and IPPO share the same actor prior, physical model and team objective.
The recurrent/attention TD3 method is an explicitly discrete adaptation, not
an exact reproduction of a continuous resource allocation algorithm. The
legacy conditional shuffle accountant is exploratory and does not establish
end-to-end or longitudinal differential privacy.

No real user data, private repository history or document sources are included.
