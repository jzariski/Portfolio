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

