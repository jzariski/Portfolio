# FLI-Point Telescope Acquisition Trainer

A Streamlit dashboard for training and validating telescope pointing offset models.
This project supports upload of text data, feature augmentation, and model evaluation for azimuth/altitude correction.

## Quickstart

```bash
# 1) Create a python virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -U pip
pip install -r requirements.txt

# 3) Launch the app
streamlit run app.py
```

## What this repository contains

```
.
├─ app.py                    # Streamlit app entrypoint for model training and prediction
├─ ModelBuilder/             # Model building and evaluation code
│  ├─ ModelBuilder.py        # XGBoost model trainer and evaluator
│  ├─ ModelBuilderNN.py      # Neural network variant of the trainer
│  └─ testBuilder.py         # Unit tests for model builder utilities
├─ Parsing/                  # Input parsing utilities for telescope files
│  ├─ TextParser.py          # Parser for log files and text data formats
│  └─ testParser.py         # Unit tests for parsing logic
├─ unix_example_small.txt    # sample input using UNIX timestamp format
├─ ymd_example_small.txt     # sample input using YYYY-MM-DD HH:MM:SS format
├─ requirements.txt          # Python package dependencies
├─ tests/                    # repository tests and integration checks
└─ .gitignore                # files and folders excluded from version control
```

## Notes for reviewers

- Designed to be clear and approachable for telescope operators.
- The model training flow includes optional feature augmentation and validation output.
- Sample data files are provided for quick local testing.

