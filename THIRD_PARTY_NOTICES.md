# Third-party notices

## TSB-AD

Parts of `src/agentad/evaluation` are adapted from
[TheDatumOrg/TSB-AD](https://github.com/TheDatumOrg/TSB-AD), including its
point, event, range, affiliation and volume-under-surface metric definitions.
TSB-AD is distributed under the Apache License 2.0.

The complete license text is retained at `forks/TSB-AD/LICENSE` and is included
in AgentAD distribution metadata through `pyproject.toml`.

AgentAD replaces the benchmark's Python loops with validated NumPy/Numba APIs,
compiled kernels and collection-level orchestration. Boundary behavior that
intentionally differs from the reference implementation is documented in
`src/agentad/evaluation/README.md`.
