Run the project test suite using uv.

Before running tests, confirm `uv sync --extra dev` has been run (dev extras include pytest).

Run the appropriate test suite based on context:

```bash
# Core game/graph tests (default)
uv run pytest tests/ -v

# Visualization tests
uv run pytest render/tests/ -v

# Single test (example)
uv run pytest tests/test_position.py::test_swap_players_positions -v
```

If no specific scope is requested, run `uv run pytest tests/ -v` and report results.
