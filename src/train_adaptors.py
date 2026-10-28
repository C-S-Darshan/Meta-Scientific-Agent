"""
train_adaptors.py

Trains dual QLoRA adapters (Hypothesis + Analyst) on a 4-bit
quantized Qwen2.5-1.5B backbone.

Literature-aligned design decisions:
  - 7 target modules incl. MLP layers (META-LORA, Beyond PEFT)
  - Warmup + cosine LR schedule (PEFT Survey)
  - Gradient clipping max_norm=1.0 (training stability)
  - 90/10 train/val split with eval_strategy (Thesis Algorithm 2)
  - Self-evolution queue integration (Data Flywheel)
  - Per-adapter hyperparameter differentiation
"""

import os
import gc
import json
import torch
import warnings
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Guard against buggy torchao version check in peft (Colab Python 3.13 image bug)
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
    if hasattr(peft.import_utils, "_torchao_available"):
        peft.import_utils._torchao_available = False
except Exception:
    pass

try:
    import peft.tuners.lora.torchao
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception:
    pass

import config

warnings.filterwarnings("ignore")


# ============================================================
# ADAPTER TRAINING QUEUE
# ============================================================
# Each entry: (adapter_name, data_path, per-adapter overrides)

TRAINING_QUEUE = [
    {
        "name": "hypothesis_adapter",
        "data": os.path.join(config.TRAIN_DATA_DIR, "hypothesis_train.jsonl"),
        "epochs": 5,
        "lr": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
    },
    {
        "name": "analyst_adapter",
        "data": os.path.join(config.TRAIN_DATA_DIR, "analyst_train.jsonl"),
        "epochs": 3,        # Analyst needs less epochs (natural language)
        "lr": 3e-4,         # Slightly higher LR for shorter training
        "lora_r": 16,
        "lora_alpha": 32,
    },
]


# ============================================================
# SELF-EVOLUTION QUEUE INTEGRATION
# ============================================================

def load_self_evolution_data(adapter_name):
    """
    Check if the self-evolution queue has accumulated entries
    from previous discovery runs. If so, convert them into
    additional training samples for the next LoRA refinement.

    Also loads corrective samples generated when proposed alloys
    failed to meet industry targets (self-evolution on failure).
    """
    queue_path = os.path.join(
        config.EXPERIMENTS_DIR,
        "self_evolution_queue.jsonl"
    )

    if not os.path.exists(queue_path):
        return []

    evo_samples = []

    with open(queue_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            if adapter_name == "hypothesis_adapter":
                # Convert logged hypothesis into training format
                text = (
                    f"Task: Propose a new alloy chemistry based on the baseline.\n"
                    f"Output ONLY the elements and percentages. Do not write anything else.\n"
                    f"Format: Element-Percentage Element-Percentage\n\n"
                    f"--- Actual Task ---\n"
                    f"Goal: {entry.get('goal', '')}\n"
                    f"Application Context: None\n"
                    f"Baseline: None\n"
                    f"Rule: None\n"
                    f"Proposed Chemistry: {entry.get('hypothesis', '')}"
                )
                evo_samples.append({"text": text})

            elif adapter_name == "analyst_adapter":
                text = (
                    f"You are an expert metallurgical analyst.\n"
                    f"Goal: {entry.get('goal', '')}\n"
                    f"Simulation Results: {{}}\n"
                    f"Conclusion (maximum 2 sentences): {entry.get('analysis', '')}"
                )
                evo_samples.append({"text": text})

    # ── Load corrective samples from failure-driven self-evolution ──
    corrections_path = os.path.join(
        config.EXPERIMENTS_DIR,
        "self_evolution_corrections.jsonl"
    )

    correction_count = 0
    if os.path.exists(corrections_path):
        target_type = "hypothesis" if adapter_name == "hypothesis_adapter" else "analyst"
        with open(corrections_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if entry.get("type") == target_type and "text" in entry:
                    evo_samples.append({"text": entry["text"]})
                    correction_count += 1

    if evo_samples:
        print(f"    [+] Loaded {len(evo_samples)} self-evolution samples from queue")
        if correction_count > 0:
            print(f"    [+]   └── {correction_count} are failure-corrective samples")

    return evo_samples


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_single_adapter(adapter_cfg):
    """Train a single LoRA adapter with literature-aligned settings."""

    name = adapter_cfg["name"]
    dataset_path = adapter_cfg["data"]

    print(f"\n{'=' * 60}")
    print(f"  TRAINING ADAPTER: {name}")
    print(f"  Base Model: {config.ACTIVE_MODEL_NAME}")
    print(f"  Epochs: {adapter_cfg['epochs']} | LR: {adapter_cfg['lr']}")
    print(f"  LoRA rank: {adapter_cfg['lora_r']} | alpha: {adapter_cfg['lora_alpha']}")
    print(f"{'=' * 60}")

    # --- Validation ---
    if not os.path.exists(config.BASE_MODEL_PATH):
        print(f"  ERROR: Model not found at {config.BASE_MODEL_PATH}")
        return

    if not os.path.exists(dataset_path):
        print(f"  ERROR: Training data not found at {dataset_path}")
        return

    # --- Load Dataset ---
    dataset = load_dataset(
        "json",
        data_files=dataset_path,
        split="train"
    )

    # Merge self-evolution queue samples
    evo_samples = load_self_evolution_data(name)
    if evo_samples:
        from datasets import Dataset
        evo_dataset = Dataset.from_list(evo_samples)
        from datasets import concatenate_datasets
        dataset = concatenate_datasets([dataset, evo_dataset])
        print(f"    [+] Total training samples: {len(dataset)}")

    # --- Train/Val Split (90/10) ---
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"    [+] Train: {len(train_dataset)} | Val: {len(eval_dataset)}")

    # --- 4-Bit Quantization Config (IRQLoRA) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,       # IRQLoRA info retention
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # --- Load Base Model ---
    is_local = os.path.isdir(config.BASE_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=is_local,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL_PATH,
        local_files_only=is_local,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- VRAM Optimizations ---
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # --- LoRA Config (META-LORA + Beyond PEFT) ---
    # Target all 7 projection layers including MLP for
    # domain-specific knowledge retention. This naming convention (q/k/v/o_proj
    # + gate/up/down_proj) matches Qwen2.5-1.5B, Gemma-2B, and
    # Llama-3.2-3B-Instruct (all Llama-style architectures), but NOT Phi-2,
    # whose attention output projection is named "dense" and whose MLP is a
    # plain 2-layer "fc1"/"fc2" (no gating) -- "all-linear" auto-detects every
    # nn.Linear layer regardless of naming, avoiding a hardcoded guess for
    # that one architecture.
    target_modules = (
        "all-linear" if config.ACTIVE_MODEL_NAME == "Phi-2"
        else ["q_proj", "k_proj", "v_proj", "o_proj",   # Attention
              "gate_proj", "up_proj", "down_proj"]        # MLP (Beyond PEFT)
    )
    peft_config = LoraConfig(
        r=adapter_cfg["lora_r"],
        lora_alpha=adapter_cfg["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=0.05,     # Lower dropout for small datasets
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    trainable, total = model.get_nb_trainable_parameters()
    print(f"    [+] Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # --- Tokenization ---
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.MAX_LENGTH,
            padding="max_length"
        )

    train_tokenized = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    eval_tokenized = eval_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    output_dir = os.path.join(config.LORA_ADAPTERS_DIR, name)

    # --- Training Arguments (Literature-aligned) ---
    training_args = TrainingArguments(
        output_dir=output_dir,

        # Batch & Accumulation
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUMULATION,

        # Epochs & LR
        num_train_epochs=adapter_cfg["epochs"],
        learning_rate=adapter_cfg["lr"],

        # Schedule (PEFT Survey: warmup + cosine)
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,

        # Stability (gradient clipping)
        max_grad_norm=1.0,

        # Precision
        fp16=True,

        # Logging & Eval
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Optimizer
        optim="paged_adamw_8bit",

        # Misc
        report_to="none",
        save_total_limit=1,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer,
            mlm=False
        ),
    )

    # --- Train ---
    print(f"\n    [*] Starting training...")
    train_result = trainer.train()

    # Log final metrics
    print(f"    [+] Final train loss: {train_result.training_loss:.4f}")

    eval_result = trainer.evaluate()
    print(f"    [+] Final eval loss: {eval_result['eval_loss']:.4f}")

    # --- Save Best Adapter ---
    model.save_pretrained(output_dir)
    print(f"    [+] Adapter saved → {output_dir}")

    # --- Nuclear Cleanup for VRAM ---
    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  META-LORA ADAPTER TRAINING PIPELINE")
    print(f"  Framework: {config.ACTIVE_MODEL_NAME}")
    print(f"  Device: {config.DEVICE}")
    print("=" * 60)

    os.makedirs(config.LORA_ADAPTERS_DIR, exist_ok=True)

    for adapter_cfg in TRAINING_QUEUE:
        train_single_adapter(adapter_cfg)

    print(f"\n{'=' * 60}")
    print(f"  ALL ADAPTERS TRAINED SUCCESSFULLY")
    print(f"  Saved to: {config.LORA_ADAPTERS_DIR}")
    print(f"{'=' * 60}")