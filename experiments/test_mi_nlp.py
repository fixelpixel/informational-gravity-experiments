"""
Phase 1 — NLP test: проверка MI-Attention на длинных текстовых зависимостях.
Время запуска: 8–12 минут (на CPU чуть дольше).
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.mi_gpt2 import inject_mi_attention

# ----------- создаём синтетическую задачу -----------

def create_memory_tasks(tokenizer, n_samples=60, gap=50):
    """
    Формат:
    "Remember this word: <key>. Now read: <random words>. What word should you remember?"
    """
    tasks = []
    filler_words = [
        "the", "a", "and", "is", "was", "were", "have", "has",
        "do", "does", "can", "could", "will", "would", "should",
        "this", "that", "there", "here", "where", "some", "many", "people"
    ]
    keys = ["apple", "bridge", "castle", "dragon", "forest", "guitar", "hammer", "island"]

    for i in range(n_samples):
        key = np.random.choice(keys)
        filler = " ".join(np.random.choice(filler_words, size=gap))
        prompt = f"Remember this word: {key}. Now read: {filler}. What word should you remember?"
        tasks.append((prompt, key))
    return tasks


# ----------- тестирование модели -----------

def evaluate_model(model, tokenizer, tasks, device="cpu"):
    model.eval().to(device)
    correct = 0
    for prompt, key in tqdm(tasks, desc="Evaluating"):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        if key.lower() in generated.lower():
            correct += 1
    return correct / len(tasks)


# ----------- основной цикл -----------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading baseline GPT-2 ...")
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # генерируем задачи
    tasks = create_memory_tasks(tokenizer, n_samples=40, gap=50)

    # baseline
    print("\nBASELINE")
    baseline_acc = evaluate_model(model, tokenizer, tasks, device)
    print(f"Accuracy: {baseline_acc:.2%}")

    # MI версии
    alphas = [0.5, 1.0, 2.0]
    results = {"baseline": baseline_acc}

    for alpha in alphas:
        print("\n" + "=" * 50)
        print(f"Testing MI-Attention (alpha={alpha})")
        print("=" * 50)

        model = GPT2LMHeadModel.from_pretrained(model_name)
        vocab_size = model.config.vocab_size
        # генерируем случайную MI-матрицу (имитация предварительно вычисленной)
        mi = torch.rand(vocab_size, vocab_size) * 0.05
        mi = (mi + mi.T) / 2  # симметрично
        model = inject_mi_attention(model, mi, alpha=alpha, layer_idx=6)

        acc = evaluate_model(model, tokenizer, tasks, device)
        results[f"alpha_{alpha}"] = acc
        print(f"Accuracy: {acc:.2%}  (Δ={acc - baseline_acc:+.2%})")

    # сводка
    print("\nSUMMARY")
    for k, v in results.items():
        delta = v - baseline_acc
        print(f"{k:10s}: {v:.2%}  (Δ={delta:+.2%})")

    Path("results").mkdir(exist_ok=True)
    import json
    json.dump(results, open("results/mi_nlp_results.json", "w"), indent=2)
    print("\nSaved to results/mi_nlp_results.json")


if __name__ == "__main__":
    main()
