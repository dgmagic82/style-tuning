"""Print the training log and the before/after table in the console (for screenshots and a quick look)."""

import json

from common import OUT_DIR

rows = [json.loads(line) for line in (OUT_DIR / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
s = json.loads((OUT_DIR / "train_summary.json").read_text(encoding="utf-8"))
print(f"Модель {s['model']}, LoRA: обучается {s['trainable_params']:,} параметров из {s['total_params']:,} "
      f"({s['trainable_params'] / s['total_params']:.1%}), адаптер {s['adapter_mb']} МБ")
print(f"Обучение на CPU: {s['train_pairs']} пар, {s['epochs']} эпох, {s['steps']} шагов, {s['train_minutes']} мин\n")
print("эпоха  loss обучения  loss на 10 отложенных отзывах")
train = {}
for r in rows:
    if "loss" in r:
        train[int(r["epoch"] - 1e-9) + 1] = r["loss"]
evals = {int(round(r["epoch"])): r["eval_loss"] for r in rows if "eval_loss" in r}  # the last epoch is logged twice
for e, loss in evals.items():
    print(f"{e:>5}  {(f'{train[e]:.3f}' if e in train else '-'):>13}  {loss:>10.3f}")
print()
report = (OUT_DIR / "eval_report.md").read_text(encoding="utf-8").splitlines()
print("\n".join(line for line in report if line.startswith("|")))
