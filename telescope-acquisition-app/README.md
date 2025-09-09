# Telescope Acquisition App (Streamlit)

A public repo for my telescope acquisition trainer app, FLI-Point. 

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
├─ ModelBuilder/             # Model architecture code (XGBoost + NN)
│  ├─ ModelBuilder.py
│  ├─ ModelBuilderNN.py
│  └─ testBuilder.py
├─ Parsing/                  # Text parsing utilities
│  ├─ TextParser.py
│  └─ testParser.py
├─ unix_example_small.txt    # sample input (UNIX timestamp format)
├─ ymd_example_small.txt     # sample input (YYYY-MM-DD HH:MM:SS)
├─ requirements.txt          # pinned dependencies
├─ tests/                    # tests to verify app is working
└─ .gitignore
```

## Next steps (optional, bite-sized)

- Add a small **"How to use"** section to `app.py` sidebar (file upload -> train -> predict).
- Save small, anonymized **sample output** (plots or a tiny CSV) to a local `.data/` folder in `.gitignore`.
- If you want CI later: add a one-file GitHub Actions workflow that just runs `pip install -r requirements.txt && pytest`.
- If you later split UI vs. logic: consider moving training/prediction code into `ModelBuilder/` functions and keep `app.py` as thin UI.
- If you add a Dockerfile: keep it tiny and copy only what you need.
- expand with unit tests for your parser or model pieces as you go.
