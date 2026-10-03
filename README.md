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

## Reproduce the context-length experiment
The default run uses sequence length 60. The report's short-versus-long comparison
uses the same training settings at lengths 10, 60, and 80. Each training run
overwrites `outputs/metrics.csv`, so archive that file immediately after each run:

```bash
# Windows PowerShell
python train.py --model all --seq_length 10
Copy-Item outputs/metrics.csv outputs/metrics_seq10.csv

python train.py --model all --seq_length 80
Copy-Item outputs/metrics.csv outputs/metrics_seq80.csv

python train.py --model all --seq_length 60
# Keep this as outputs/metrics.csv for the normal app comparison.

python plot_dependency.py
```

On macOS/Linux, replace the two `Copy-Item` commands with:

```bash
cp outputs/metrics.csv outputs/metrics_seq10.csv
cp outputs/metrics.csv outputs/metrics_seq80.csv
```

Run the 60-character experiment before `plot_dependency.py` so that the script
can read `outputs/metrics.csv`. The plotting script combines the archived files
into `outputs/dependency_metrics.csv` and creates
`outputs/dependency_accuracy.png`, which are used by the report and the app.
The model files and per-model history files are also overwritten on each run;
archive those separately if you need the trained models or learning curves for
all three sequence lengths.

This is a limited comparison on one dataset and training setup. The results do
not establish that Simple RNN accuracy consistently decreases with context
length, or that gated models are universally better for long-range dependencies.

## Interpreting generated text
Use the fixed-prompt comparison in the app to compare RNN, LSTM, and GRU at the
same temperature. These are character-level next-character models trained on a
small corpus, so occasional incoherent text is expected; validation accuracy
around 38--40% measures the next-character task and does not guarantee fluent
multi-character generation.

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
