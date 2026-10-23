"""Console chat with the fine-tuned model, as in the lesson: type a review, get the store's reply.

    uv run python chat.py          # with the LoRA adapter
    uv run python chat.py --base   # the base model, for comparison

Runs fully offline once the model is cached in ./models: reviews never leave the computer.
"""

import sys

from peft import PeftModel

from common import ADAPTER_DIR, drop_contacts, generate, load_base


def main() -> None:
    base_only = "--base" in sys.argv
    print("Загружаю модель…", flush=True)
    tokenizer, model = load_base()
    if not base_only:
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    print(f"Готово ({'базовая модель' if base_only else 'с адаптером стиля «Тёплого дома»'}). "
          "Вставьте отзыв покупателя, пустая строка или «выход» для выхода.\n")
    seed = 1
    while True:
        try:
            review = input("Отзыв> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not sys.stdin.isatty():  # piped demo input is not echoed by the console, show it
            print(review)
        if not review or review.lower() in {"выход", "exit", "quit"}:
            break
        print("\nОтвет:\n" + drop_contacts(generate(model, tokenizer, review, seed=seed)) + "\n", flush=True)
        seed += 1


if __name__ == "__main__":
    main()
