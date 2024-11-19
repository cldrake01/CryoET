# CryoET

See [CZII - CryoET Object Identification](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/overview).

## Poetry

Poetry can be loaded with `poetry update`.

For Zed, Poetry can be recognized by its LSP with the following command:

```python
poetry env info -p | read -r d; printf '{\n  "venvPath": "%s",\n  "venv": "%s"\n}\n' "$(dirname "$d")" "$(basename "$d")" > pyrightconfig.json
```

Also, paste the following into your `settings.json`:

```json
"lsp": {
  // Python
  "pyright": {
    "settings": {
      "python.analysis": {
        "diagnosticMode": "workspace",
        "typeCheckingMode": "strict"
      },
      "python": {
        "pythonPath": ".venv/bin/python"
      }
    }
  }
},
```
