# IDS MoE + (Optional) Local LLM on NSLKDD

A small Intrusion Detection System (IDS) demo using the **NSLKDD** dataset.

It trains a **Mixture-of-Experts (MoE)** classifier (Random Forest + Logistic Regression + SVM) to do **binary detection**:
- `0` = normal traffic
- `1` = attack

Then it optionally adds an **LLM review/explanation step**:
- **Offline mode** (always works): a *simulated* LLM that applies a few readable heuristics and produces explanations.
- **Real mode**: uses a **local Ollama model** (e.g., `phi3` or `tinyllama`) to produce a YES/NO decision + one-line explanation.

## Project files

| File | What it does |
|------|--------------|
| `ids_moe_llm.py` | MoE baseline + **simulated (offline) LLM reasoning**. Writes `llm_explanations.json`. |
| `ids_moe_llm_real.py` | MoE baseline + **real local LLM via Ollama** (with fallbacks). Writes `real_llm_results.json`. |
| `download_data.py` | Downloads `KDDTrain+.txt` and `KDDTest+.txt` into the repo root. |
| `KDDTrain+.txt`, `KDDTest+.txt` | NSLKDD train/test files (expected in the repo root). |

## Requirements

- Python 3.9+ (3.10+ recommended)
- Core packages:
  - `pandas`, `numpy`, `scikit-learn`
- Optional (only for **real** LLM mode):
  - `ollama` Python package
  - Ollama installed and running locally

### Optional: Ollama setup

1. Install Ollama from: https://ollama.com/
2. Pull a model (example):
   - `ollama pull phi3`
   - or `ollama pull tinyllama`

`ids_moe_llm_real.py` tries `phi3` first and falls back to `tinyllama`.

## Setup (Windows / PowerShell)

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas numpy scikit-learn
```

For **real LLM mode** (Ollama):

```powershell
pip install ollama
```

## Usage

### 1) Offline MoE + simulated LLM (recommended first run)

```powershell
python .\ids_moe_llm.py
```

What youll see:
- dataset download (if missing)
- preprocessing
- MoE training
- final accuracy + F1
- a few explanation examples

Output file:
- `llm_explanations.json` (saves a few example explanations)

### 2) MoE + **real** local LLM (Ollama)

```powershell
python .\ids_moe_llm_real.py
```

Output file:
- `real_llm_results.json` (metrics + per-sample explanations + LLM latency)

Note: for speed, the script runs LLM inference only on a small sample (`n_samples`), then fills the rest with MoE predictions.

## Configuration knobs (quick edits)

### `ids_moe_llm.py`
- Low-confidence cutoff for LLM review: `moe_conf < 0.65` (inside `simulated_llm_enhance`)
- Number of test samples to review: `np.random.choice(..., 100, ...)`

### `ids_moe_llm_real.py`
- Number of samples sent to the LLM: `n_samples = 20`
- Ollama generation options in `ollama.chat(...)`:
  - `num_predict` (max tokens)
  - `temperature`

Theres also a commented block to run the LLM **only when MoE confidence is low** (useful if Ollama inference is slow).

## Troubleshooting

### Dataset download fails (SSL / corporate proxy)
The scripts already include a Windows SSL bypass and a manual-download fallback.
You can also download the files manually and place them in the repo root:
- https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt
- https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt

### Ollama errors (real mode)
Common causes:
- Ollama isnt installed or not running
- the model isnt pulled yet (`ollama pull phi3`)

If Ollama isnt available, `ids_moe_llm_real.py` attempts fallbacks (Transformers-based, then rule-based heuristics).

### Slow runtime
- Random Forest + SVM training can take time.
- Real LLM inference is the slowest part. Reduce `n_samples` or enable the only low-confidence LLM block.

## Notes

- This is a research/demo prototype, not a production IDS.
- The MoE/LLM step is intended to add interpretability (and sometimes correct low-confidence cases), but results depend on sampling and model availability.

## Dataset credit

This project uses the NSLKDD dataset (downloaded from the public mirror used in the scripts):
https://github.com/defcom17/NSL_KDD
