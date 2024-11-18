# CryoET

See [CZII - CryoET Object Identification](https://www.kaggle.com/competitions/czii-cryo-et-object-identification/overview).

## Poetry

Poetry can be loaded with `poetry update`.

For Zed, Poetry can be recognized by its LSP with the following command:

```
poetry env info -p | read -r d; printf '{\n  "venvPath": "%s",\n  "venv": "%s"\n}\n' "$(dirname "$d")" "$(basename "$d")" > pyrightconfig.json
```
