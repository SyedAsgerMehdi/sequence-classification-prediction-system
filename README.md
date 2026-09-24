# Sequence Prediction System — College Ready Project

## What this project does
A complete character-level sequence prediction system that predicts the next character and compares Simple RNN, LSTM and GRU.

## Run
```bash
pip install -r requirements.txt
python train.py --model all
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

## Generate from terminal
```bash
python generate.py --model lstm --seed "ROMEO: "
```

## Dataset
The training script automatically downloads Tiny Shakespeare. If download fails, it uses `data/sample.txt`.

## Deliverables
- `train.py` — model training
- `generate.py` — sequence generation
- `app.py` — interactive web interface
- `utils.py` — preprocessing/prediction utilities
- `PROJECT_REPORT.md` — report content
- `requirements.txt` — dependencies
- `data/sample.txt` — fallback dataset
- `outputs/` — generated models, metrics and graphs

## Important
For sequential text, the correct standard term is **Recurrent Neural Network (RNN)**. "Recursive Neural Network" is a different architecture.
