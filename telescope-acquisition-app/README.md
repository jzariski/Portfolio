# Telescope Acquisition App (Streamlit)

Minimal, runnable repo for your telescope acquisition trainer app. This wraps your existing code with a clean layout, a `requirements.txt`, a `.gitignore`, and a tiny test.

## Quickstart

```bash
# 1) Create a virtual env (recommended)
python -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -U pip
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

## Project layout

```
.
├─ app.py                    # Streamlit app entrypoint
├─ ModelBuilder/             # Your model code (XGBoost + NN)
│  ├─ ModelBuilder.py
│  ├─ ModelBuilderNN.py
│  └─ testBuilder.py
├─ Parsing/                  # Your text parsing utilities
│  ├─ TextParser.py
│  └─ testParser.py
├─ unix_example_small.txt    # sample input (UNIX timestamp format)
├─ ymd_example_small.txt     # sample input (YYYY-MM-DD HH:MM:SS)
├─ requirements.txt          # pinned dependencies (editable)
├─ tests/                    # minimal test to verify imports work
└─ .gitignore
```

## Next steps (optional, bite-sized)

- Add a small **"How to use"** section to `app.py` sidebar (file upload -> train -> predict).
- Save small, anonymized **sample output** (plots or a tiny CSV) to a local `.data/` folder in `.gitignore`.
- If you want CI later: add a one-file GitHub Actions workflow that just runs `pip install -r requirements.txt && pytest`.
- If you later split UI vs. logic: consider moving training/prediction code into `ModelBuilder/` functions and keep `app.py` as thin UI.
- If you add a Dockerfile: keep it tiny and copy only what you need.

## Notes

- This repo intentionally stays **minimal** (no heavy restructuring) so your current imports keep working.
- The sample tests only confirm imports; expand with unit tests for your parser or model pieces as you go.
