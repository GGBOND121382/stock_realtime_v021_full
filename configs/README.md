# saved_configs

This folder keeps reproducible command configs for known stock workflows.

Run from repository root:

```powershell
D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe stock_realtime\run_saved_config.py list
D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe stock_realtime\run_saved_config.py show 600312_pipeline
D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe stock_realtime\run_saved_config.py run 600312_pipeline --dry-run
```

The configs are TOML files. They may contain comments, and placeholders are
expanded by `run_saved_config.py`:

- `{python}`: current Python interpreter, or `--python` override.
- `{project_dir}`: absolute path to `stock_realtime`.
- `{saved_data}`: absolute path to `stock_realtime/saved_data`.
- `{saved_models}`: absolute path to `stock_realtime/saved_models`.

Use `--command <id>` to run one command from a config that contains multiple
commands. The default is `all`.
