Install/sync the project's uv dependencies including dev extras (pytest).

Run the following in the project root:

```bash
uv sync --extra dev
```

After running, confirm success by checking that pytest is available:

```bash
uv run pytest --version
```

If `.env` does not exist, remind the user to copy `.env.template` to `.env` and set `GRAPH_FILES_DIR` to the path of `Graph/files/`.
