"""Before/after comparison on the 10 held-out reviews: the same base model with and without the LoRA adapter.

    uv run python evaluate.py

The style checks are computed by code, not judged by a model, and the full answers go to out/eval_report.md
so they can be read by eye.
"""

import json
import re

from peft import PeftModel

from common import ADAPTER_DIR, CONTACT, DATA_DIR, OUT_DIR, SIGNATURE, drop_contacts, generate, load_base

BANNED = ["уважаем", "данный", "данного", "в связи с", "приносим извинения", "доставленные неудобства",
          "обращаем ваше внимание", "в кратчайшие сроки"]

CHECKS = {
    "подпись бренда": lambda a, ref: a.rstrip().endswith(SIGNATURE),
    "на «вы»": lambda a, ref: bool(re.search(r"\b(вы|вас|вам|ваш\w*)\b", a.lower())),
    "без канцелярита": lambda a, ref: not any(b in a.lower() for b in BANNED),
    "не длиннее 60 слов": lambda a, ref: 0 < len(a.split()) <= 60,
    "связный русский текст": lambda a, ref: len(a) > 20 and sum(c.isalpha() and "а" <= c.lower() <= "я" or c in "ёЁ"
                                                                for c in a) / max(1, sum(c.isalpha() for c in a)) > 0.9,
    # the riskiest failure for a business: a made-up phone number, email or link in a public reply
    "нет выдуманных контактов": lambda a, ref: not CONTACT.search(a),
    # only for complaints (the reference reply apologises): an apology and a concrete next step
    "жалоба: извинение и шаг": lambda a, ref: None if not re.search(r"извин|жаль|простите", ref.lower()) else
        bool(re.search(r"извин|жаль|простите", a.lower()) and re.search(r"напишите|пришлите|замен|верн|отправ", a.lower())),
}


def score(answer: str, ref: str) -> dict:
    return {name: fn(answer, ref) for name, fn in CHECKS.items()}


def main() -> None:
    test = json.loads((DATA_DIR / "test.json").read_text(encoding="utf-8"))
    tokenizer, model = load_base()
    # one seed per review: the same seed for all of them made every answer start with the same token
    base = [generate(model, tokenizer, p["instruction"], seed=100 + i) for i, p in enumerate(test)]
    print("базовая модель готова", flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    tuned = [generate(model, tokenizer, p["instruction"], seed=100 + i) for i, p in enumerate(test)]
    print("дообученная модель готова", flush=True)

    rows = [{"review": p["instruction"], "reference": p["output"], "base": b, "tuned": t, "guarded": drop_contacts(t),
             "base_checks": score(b, p["output"]), "tuned_checks": score(t, p["output"]),
             "guarded_checks": score(drop_contacts(t), p["output"])}
            for p, b, t in zip(test, base, tuned)]
    (OUT_DIR / "eval_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def rate(key, name):
        vals = [r[key][name] for r in rows if r[key][name] is not None]
        return f"{sum(vals)}/{len(vals)}"

    lines = ["# До и после дообучения: 10 отложенных отзывов", "",
             "Модель `ai-forever/rugpt3small_based_on_gpt2`, одна и та же генерация (свой seed на каждый отзыв, temperature 0.7) "
             "без адаптера и с LoRA-адаптером. Проверки считает код (`evaluate.py`).", "",
             "Колонка «После + фильтр»: тот же ответ после `drop_contacts`, который убирает предложения с телефонами, "
             "почтой и ссылками. Это защита в коде, а не заслуга обучения.", "",
             "| Проверка | До | После | После + фильтр |", "|---|---|---|---|"]
    lines += [f"| {name} | {rate('base_checks', name)} | {rate('tuned_checks', name)} | "
              f"{rate('guarded_checks', name)} |" for name in CHECKS]
    lines += ["", "## Ответы", ""]
    for i, r in enumerate(rows, 1):
        lines += [f"### {i}. {r['review']}", "", f"**До:** {r['base'] or '(пусто)'}", "",
                  f"**После:** {r['tuned'] or '(пусто)'}", "",
                  *([f"**После + фильтр:** {r['guarded']}", ""] if r["guarded"] != r["tuned"] else []),
                  f"**Эталон из датасета:** {r['reference']}", ""]
    (OUT_DIR / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8 + len(CHECKS)]))


if __name__ == "__main__":
    main()
