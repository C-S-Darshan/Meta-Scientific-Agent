import os
import torch

# --- BASE PATHS ---
# Assuming this file is inside Meta-Scientific-Agent/src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- DATA DIRECTORIES ---
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")

# --- MODEL & ADAPTER PATHS ---
MODELS_DIR = os.path.join(BASE_DIR, "models")
LORA_ADAPTERS_DIR = os.path.join(BASE_DIR, "lora_adapters")

HYPOTHESIS_ADAPTER_DIR = os.path.join(LORA_ADAPTERS_DIR, "hypothesis_adapter")
ANALYST_ADAPTER_DIR = os.path.join(LORA_ADAPTERS_DIR, "analyst_adapter")

# --- EXPERIMENTS & PHYSICS PATHS ---
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
GRAPHS_DIR = os.path.join(EXPERIMENTS_DIR, "graphs")
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")
TDB_DIR = os.path.join(BASE_DIR, "thermodynamic_databases")

# Universal Model Mapping: Fallback to HuggingFace Hub if local directory does not exist
# Uses 100% freely downloadable, ungated open-source checkpoints (no HF token required)
MODEL_HUB_MAP = {
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Phi-2": "microsoft/phi-2",
    "Phi-2-2.7B": "microsoft/phi-2",
    "Gemma-2B": "unsloth/gemma-2-2b-it",
    "Gemma-2-2B": "unsloth/gemma-2-2b-it",
    "Llama-3.2-1B": "unsloth/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B": "unsloth/Llama-3.2-3B-Instruct",
}

ACTIVE_MODEL_NAME = os.getenv("ACTIVE_MODEL_NAME", "Qwen2.5-1.5B")
_local_model_path = os.path.join(MODELS_DIR, ACTIVE_MODEL_NAME)
if os.path.exists(_local_model_path):
    BASE_MODEL_PATH = _local_model_path
else:
    BASE_MODEL_PATH = MODEL_HUB_MAP.get(ACTIVE_MODEL_NAME, "Qwen/Qwen2.5-1.5B-Instruct")

# --- DISTRIBUTED NETWORK CONFIG ---
# The coordinator will iterate through these manually provided worker IPs to establish connections.
MANUAL_WORKER_IPS = [
    # "100.x.x.x",  # e.g., Tailscale IP for colab-worker
]
COORDINATOR_PORT = 5000

# --- HARDWARE & TRAINING SETTINGS ---
BATCH_SIZE = 1           # Optimized for consumer hardware VRAM limits
MAX_TOKEN_LENGTH = 512   # Max context for inference
MAX_LENGTH = 256         # Max context for training (increased to prevent truncation)
GRAD_ACCUMULATION = 4    # Simulates larger batches
LEARNING_RATE = 2e-4

# --- DEVICE DETECTION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_INDEX = 0

# Optional initialization feedback
if __name__ == "__main__":
    print(f"--- Meta-Scientific Agent Config Initialized ---")
    print(f"Active Framework: {ACTIVE_MODEL_NAME}")
    print(f"Base Directory: {BASE_DIR}")
    if DEVICE == "cuda":
        print(f"✓ GPU detected: {torch.cuda.get_device_name(DEVICE_INDEX)}")
    else:
        print("⚠ GPU not available, using CPU")