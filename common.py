"""Shared bits: model id, paths, prompt format, model loading and generation."""

import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_ID = "ai-forever/rugpt3small_based_on_gpt2"  # the lesson's sberbank-ai/rugpt3small, renamed on Hugging Face
MODELS_DIR = ROOT / "models"                       # local cache, so nothing is fetched again
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
ADAPTER_DIR = OUT_DIR / "adapter"
SIGNATURE = "Команда «Тёплого дома»"

torch.set_num_threads(8)  # leave some cores free: the laptop stays usable during training


def prompt(review: str) -> str:
    return f"Отзыв покупателя:\n{review.strip()}\n\nОтвет магазина:\n"


def load_base():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODELS_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=MODELS_DIR, dtype=torch.float32)
    return tokenizer, model


CONTACT = re.compile(r"\d[\d\s()-]{6,}\d|\S+@\S+|https?://\S+|www\.\S+|\b\S+\.ru\b")


def drop_contacts(reply: str) -> str:
    """Guard for drafts: the model sometimes invents phone numbers and emails (from pretraining, not from our data);
    a public reply must never carry them, so sentences with contacts are removed."""
    body, sep, signature = reply.rpartition(SIGNATURE)
    if not sep:
        body, signature = reply, ""
    sentences = re.split(r"(?<=[.!?])\s+", body.strip())
    kept = " ".join(s for s in sentences if not CONTACT.search(s))
    return (kept + ("\n" + SIGNATURE + signature if sep else "")).strip()


def generate(model, tokenizer, review: str, seed: int = 42, max_new_tokens: int = 110) -> str:
    torch.manual_seed(seed)
    inputs = tokenizer(prompt(review), return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                             repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # a small model keeps writing the next "review"; the answer ends at the signature or the next prompt header
    text = re.split(r"\n\s*\n|Отзыв покупателя:", text)[0] if SIGNATURE not in text else \
        text[:text.index(SIGNATURE) + len(SIGNATURE)]
    return text.strip()
