# CryoET

See [CZII - CryoET Object Identification](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/overview).

## Poetry

Poetry can be initialized with `poetry init`. The `pyproject.toml` file can be updated with the following command:

```
poetry add --dev black flake8 isort mypy pylint pytest pytest-cov pytest-mock
```

For Zed, Poetry can be recognized by its LSP with the following command:

```
poetry env info -p | read -r d; printf '{\n  "venvPath": "%s",\n  "venv": "%s"\n}\n' "$(dirname "$d")" "$(basename "$d")" > pyrightconfig.json
```


