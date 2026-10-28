# Meta-Scientific Agent

A Neuro-Symbolic Multi-Agent Framework for Computational Alloy Design.

The Meta-Scientific Agent integrates language-model hypothesis generation, deterministic composition validation, empirical property estimation, and CALPHAD thermodynamic screening into an end-to-end materials discovery pipeline.

---

## Architecture Overview

1. **Hypothesis Generation & Reasoning (`src/agent.py`)**  
   Coordinates reasoning across role-specific adapters over an open-weights SLM backbone (e.g. Qwen2.5 / Phi / Gemma).
2. **Deterministic Stoichiometry & Symbolic Physics (`src/formulas.py`, `src/validate_symbolic.py`)**  
   Enforces exact mass conservation, PREN (Pitting Resistance Equivalent Number), elemental density calculations, and empirical rule-based boundary checks.
3. **CALPHAD Thermodynamic Verification (`src/thermodynamic_test.py`, `thermodynamic_databases/`)**  
   Interfaces with PyCalphad and thermodynamic databases (`.tdb`) to assess equilibrium phase stability and formation properties.
4. **Interactive Web Interface (`web_ui/`) & API Backend (`src/api/server.py`)**  
   FastAPI service exposing REST endpoints paired with a modern Vite/React user interface.

---

## Getting Started

### 1. Installation

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/C-S-Darshan/Meta-Scientific-Agent.git
cd Meta-Scientific-Agent

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Running the Backend Server

Start the FastAPI application:

```bash
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

The interactive API documentation is available at `http://localhost:8000/docs`.

### 3. Running the Frontend UI

From the `web_ui` folder:

```bash
cd web_ui
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Directory Structure

```
Meta-Scientific-Agent/
├── src/                          # Core agent, API, and physics routines
│   ├── agent.py                  # Agent reasoning loop & LoRA orchestration
│   ├── config.py                 # System configuration & model mappings
│   ├── formulas.py               # Deterministic stoichiometry & metallurgy formulas
│   ├── thermodynamic_test.py     # CALPHAD thermodynamic assessment
│   ├── validate_symbolic.py      # Rule-based validation filters
│   └── api/
│       └── server.py             # FastAPI backend server
├── thermodynamic_databases/      # CALPHAD TDB database files
├── web_ui/                       # React + Vite interactive frontend
├── tests/                        # Unit tests for formulas & validation
├── requirements.txt              # Python dependencies
└── README.md
```

---

## License

This project is released under the MIT License.
