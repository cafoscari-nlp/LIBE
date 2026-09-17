"""
Train a sentence-transformer bi-encoder for language identification.

Each training example pairs:
  anchor: an input text segment
  positive: either a language label or a natural-language language description.

The model is optimized with MultipleNegativesRankingLoss.
"""

import argparse
import csv
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from huggingface_hub import login
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    models,
    datasets,
)


def is_gemma_model(model_name: str) -> bool:
    return "gemma" in model_name.lower()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True


def require_file(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_language_descriptions(description_variant: str, resource_dir: str | Path) -> dict[str, str]:
    resource_dir = Path(resource_dir)

    if description_variant == "summarized_50":
        path = resource_dir / "WorldsLangs_Summarized_50.tsv"
    elif "summ" in description_variant:
        path = resource_dir / "WorldsLangs_Summarized.tsv"
    else:
        path = resource_dir / "WorldsLangs.tsv"

    path = require_file(path)

    language_descriptions = {}

    with path.open(encoding="utf-8") as fr:
        reader = csv.reader(fr, delimiter="\t")

        for row in reader:
            if len(row) < 2:
                continue

            description = row[1].replace("\xa0", " ")
            description = re.sub(r"\[\d+\]", "", description)
            description = re.sub(r"\s+", " ", description)
            description = description.replace("\t", " ")

            language_descriptions[row[0]] = description

    if description_variant == "shuffled":
        keys = list(language_descriptions.keys())
        values = list(language_descriptions.values())
        random.shuffle(keys)
        random.shuffle(values)
        language_descriptions = dict(zip(keys, values))

    return language_descriptions


def format_example(
    lang: str,
    anchor: str,
    language_descriptions: dict[str, str],
    model_name: str,
    positive_type: str,
) -> InputExample | None:
    lang_clean = lang.replace("__label__", "").split("_")[0]

    if positive_type == "label":
        positive = lang
    else:
        if lang_clean not in language_descriptions:
            return None
        positive = language_descriptions[lang_clean]

    if "multilingual-e5" in model_name:
        anchor = "query: " + anchor
        positive = "passage: " + positive

    return InputExample(texts=[anchor, positive])


def sample_examples_by_index(
    file_path: str | Path,
    language_descriptions: dict[str, str],
    model_name: str,
    positive_type: str,
    sample_n: int,
    seed: int,
    temperature: float,
) -> list[InputExample]:
    rng = random.Random(seed)
    file_path = require_file(file_path)

    indices_by_lang = defaultdict(list)

    with file_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()

            if not line:
                continue

            try:
                lang, _ = line.split("\t", 1)
            except ValueError:
                continue

            lang_clean = lang.replace("__label__", "").split("_")[0]

            if positive_type != "label" and lang_clean not in language_descriptions:
                continue

            indices_by_lang[lang_clean].append(i)

    if not indices_by_lang:
        return []

    class_counts = {lang: len(indices) for lang, indices in indices_by_lang.items()}

    class_weights = {
        lang: count * ((1.0 / count) ** temperature)
        for lang, count in class_counts.items()
    }

    weight_sum = sum(class_weights.values())

    target_per_lang = {
        lang: int(round(sample_n * weight / weight_sum))
        for lang, weight in class_weights.items()
    }

    sampled_indices = []

    for lang, target_n in target_per_lang.items():
        available = indices_by_lang[lang]

        if target_n <= len(available):
            sampled_indices.extend(rng.sample(available, target_n))
        else:
            sampled_indices.extend(available)
            sampled_indices.extend(rng.choices(available, k=target_n - len(available)))

    if len(sampled_indices) > sample_n:
        sampled_indices = rng.sample(sampled_indices, sample_n)
    elif len(sampled_indices) < sample_n:
        all_valid_indices = [
            i
            for indices in indices_by_lang.values()
            for i in indices
        ]
        sampled_indices.extend(
            rng.choices(all_valid_indices, k=sample_n - len(sampled_indices))
        )

    selected_counts = Counter(sampled_indices)
    examples_by_index = {}

    with file_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i not in selected_counts:
                continue

            line = line.strip()

            if not line:
                continue

            try:
                lang, anchor = line.split("\t", 1)
            except ValueError:
                continue

            example = format_example(
                lang=lang,
                anchor=anchor,
                language_descriptions=language_descriptions,
                model_name=model_name,
                positive_type=positive_type,
            )

            if example is not None:
                examples_by_index[i] = example

    train_examples = []

    for i, count in selected_counts.items():
        if i in examples_by_index:
            train_examples.extend([examples_by_index[i]] * count)

    rng.shuffle(train_examples)

    return train_examples


def load_full_dataset(
    file_path: str | Path,
    language_descriptions: dict[str, str],
    model_name: str,
    positive_type: str,
) -> list[InputExample]:
    file_path = require_file(file_path)
    train_examples = []

    with file_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                lang, anchor = line.split("\t", 1)
            except ValueError:
                continue

            example = format_example(
                lang=lang,
                anchor=anchor,
                language_descriptions=language_descriptions,
                model_name=model_name,
                positive_type=positive_type,
            )

            if example is not None:
                train_examples.append(example)

    return train_examples


def build_sentence_transformer(model_name: str) -> SentenceTransformer:
    if is_gemma_model(model_name):
        print(
            "[INFO] Gemma detected: BF16 weights, max_seq_length=512, AMP disabled",
            flush=True,
        )

        word_embeddings = models.Transformer(
            model_name,
            max_seq_length=512,
            model_args={"torch_dtype": torch.bfloat16},
            tokenizer_args={"model_max_length": 512},
        )
    else:
        word_embeddings = models.Transformer(model_name)

    pooling = models.Pooling(word_embeddings.get_word_embedding_dimension())

    return SentenceTransformer(modules=[word_embeddings, pooling])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--description", default="full")
    parser.add_argument("--positive-type", choices=["desc", "label"], default="desc")
    parser.add_argument("--sample-n", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resource-dir", default="resources")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--use-full-dataset", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    set_seed(args.seed)
    configure_torch()

    language_descriptions = load_language_descriptions(
        description_variant=args.description,
        resource_dir=args.resource_dir,
    )

    if args.use_full_dataset:
        print(f"[INFO] Using full dataset: {args.dataset}", flush=True)
        train_examples = load_full_dataset(
            file_path=args.dataset,
            language_descriptions=language_descriptions,
            model_name=args.model,
            positive_type=args.positive_type,
        )
        sampling_label = "full"
    else:
        train_examples = sample_examples_by_index(
            file_path=args.dataset,
            language_descriptions=language_descriptions,
            model_name=args.model,
            positive_type=args.positive_type,
            sample_n=args.sample_n,
            seed=args.seed,
            temperature=args.temperature,
        )
        sampling_label = f"sample-{args.sample_n}_temperature-{args.temperature}"

    print(f"Dataset loaded. Number of examples: {len(train_examples)}", flush=True)

    model_short_name = args.model.split("/")[-1]
    dataset_name = Path(args.dataset).name.replace(".tsv", "")

    final_dir = Path(args.output_dir) / (
        f"{model_short_name}_{dataset_name}_{len(train_examples)}_"
        f"{args.description}_{args.positive_type}_{sampling_label}_biencoder"
    )

    train_dataloader = datasets.NoDuplicatesDataLoader(
        train_examples,
        batch_size=args.batch_size,
    )

    model = build_sentence_transformer(args.model)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)
    gemma = is_gemma_model(args.model)

    print(
        f"Training: model={args.model}, batch_size={args.batch_size}, "
        f"epochs={args.epochs}, use_amp={not gemma}, output={final_dir}",
        flush=True,
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        save_best_model=True,
        use_amp=False if gemma else True,
        output_path=str(final_dir),
    )


if __name__ == "__main__":
    main()