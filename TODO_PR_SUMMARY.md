# Proposed Enhancements and Fixes

## Code Improvements:
1. Add logging or tracking for temperature annealing progress during training.
2. Expand diagnostics to include more detailed per-layer performance metrics for flows.
3. Document `pyproject.toml` optional dependencies and their usage.

## Tests:
1. Add edge case tests for flow depths 0 and 10.
2. Test performance under GPU execution for consistency.
3. Add ablation study tests to verify differences between MADE-only and coupled models.

## Future Directions:
1. Automate benchmarks for results at critical temperature `T_c` in `docs/` via CI script.
2. Analyze scalability to larger systems (e.g., 32x32 grids).