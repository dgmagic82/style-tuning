"""LoRA fine-tuning of ruGPT-3 small on store replies to reviews, on CPU.

    uv run python train.py

Follows the lesson's algorithm (dataset of instruction/output pairs → tokenize → LoRA r=16, alpha=32, dropout=0.05
on c_attn, c_proj, c_fc → Trainer → save the adapter) with two fixes:
- loss is computed on the reply only, the review part of the prompt is masked out (-100);
- warmup is 10% of the steps; the lesson's 100 warmup steps exceed the whole run (~17 steps) and keep LR near zero.
Every step and the per-epoch loss on the held-out test set go to out/train_log.jsonl.
"""

import json
import time

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments

from common import ADAPTER_DIR, DATA_DIR, OUT_DIR, load_base, prompt

MAX_LEN = 256
EPOCHS = 8


def encode(tokenizer, pair: dict) -> dict:
    p = tokenizer(prompt(pair["instruction"]))["input_ids"]
    a = tokenizer(pair["output"].strip() + tokenizer.eos_token)["input_ids"]
    ids = (p + a)[:MAX_LEN]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ([-100] * len(p) + a)[:MAX_LEN]}


class JsonlLog(TrainerCallback):
    def __init__(self, path):
        self.path = path
        path.write_text("", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        row = {"step": state.global_step, "epoch": round(state.epoch or 0, 2), **(logs or {})}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if "loss" in row or "eval_loss" in row:
            print(f"шаг {row['step']:>3} эпоха {row['epoch']:>4}  "
                  + "  ".join(f"{k}={row[k]:.3f}" for k in ("loss", "eval_loss") if k in row), flush=True)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    train = json.loads((DATA_DIR / "train.json").read_text(encoding="utf-8"))
    test = json.loads((DATA_DIR / "test.json").read_text(encoding="utf-8"))
    tokenizer, model = load_base()

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["c_attn", "c_proj", "c_fc"],
                      fan_in_fan_out=True,  # GPT-2 stores these layers as Conv1D, weights transposed
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"обучаемых параметров: {trainable:,} из {total:,} ({trainable / total:.2%})")

    ds_train = Dataset.from_list([encode(tokenizer, p) for p in train])
    ds_test = Dataset.from_list([encode(tokenizer, p) for p in test])
    args = TrainingArguments(output_dir=str(OUT_DIR / "checkpoints"), num_train_epochs=EPOCHS,
                             per_device_train_batch_size=8, per_device_eval_batch_size=8, learning_rate=3e-4,
                             warmup_steps=0.1, lr_scheduler_type="cosine", weight_decay=0.0, logging_steps=1,
                             eval_strategy="epoch", save_strategy="no", report_to=[], use_cpu=True, seed=42,
                             disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_test,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, padding=True),
                      callbacks=[JsonlLog(OUT_DIR / "train_log.jsonl")])

    before = trainer.evaluate()["eval_loss"]
    print(f"loss на отложенных 10 отзывах до обучения: {before:.3f}")
    started = time.time()
    trainer.train()
    after = trainer.evaluate()["eval_loss"]
    minutes = (time.time() - started) / 60
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    summary = {"model": "ai-forever/rugpt3small_based_on_gpt2", "train_pairs": len(train), "test_pairs": len(test),
               "epochs": EPOCHS, "steps": trainer.state.global_step, "trainable_params": trainable,
               "total_params": total, "eval_loss_before": round(before, 3), "eval_loss_after": round(after, 3),
               "train_minutes": round(minutes, 1),
               "adapter_mb": round(sum(f.stat().st_size for f in ADAPTER_DIR.glob("*.safetensors")) / 1e6, 1)}
    (OUT_DIR / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
