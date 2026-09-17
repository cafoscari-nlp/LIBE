"""
Train a cross-encoder for language identification reranking.

Pipeline:
  1. Load text/language-label training rows.
  2. Optionally sample rows while preserving rare languages.
  3. Optionally apply median-centered temperature sampling.
  4. Export the underlying Hugging Face transformer from a trained bi-encoder.
  5. Use the bi-encoder to mine hard negative label descriptions.
  6. Train a binary cross-encoder on positive and hard-negative pairs.

Each cross-encoder example is:
  input:  (text segment, language description)
  label:  1 for the gold language, 0 for a hard negative language
"""

import argparse
import csv
import math
import os
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from sentence_transformers import InputExample, SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_description(description: str) -> str:
    description = description.replace("\xa0", " ")
    description = re.sub(r"\[\d+\]", "", description)
    description = re.sub(r"\s+", " ", description)
    description = description.replace("\t", " ")
    return description.strip()


def normalize_label(label: str) -> str:
    """
    Convert labels such as '__label__eng_Latn' or 'eng_Latn' to 'eng'.

    This matches the convention used by your bi-encoder script.
    """
    label = label.strip()
    label = label.replace("__label__", "")
    return label.split("_")[0]


def load_language_descriptions(path: str | Path) -> dict[str, str]:
    path = require_file(path)

    print(f"Loading language descriptions from {path}", flush=True)

    descriptions = {}

    with path.open(encoding="utf-8") as fr:
        reader = csv.reader(fr, delimiter="\t")

        for row in reader:
            if len(row) < 2:
                continue

            lang_id = row[0].strip()
            descriptions[lang_id] = clean_description(row[1])

    return descriptions


def parse_train_file_spec(spec: str) -> tuple[Path, str]:
    """
    Parse a file specification of the form:

      path/to/file.tsv:tsv_label_first
      path/to/file.tsv:fasttext

    Supported formats:
      - tsv_label_first: label<TAB>text
      - fasttext: text __label__label
    """
    if ":" not in spec:
        raise ValueError(
            "Train file specs must have the form PATH:FORMAT. "
            "Supported formats: tsv_label_first, fasttext."
        )

    path_str, fmt = spec.rsplit(":", 1)
    path = require_file(path_str)

    allowed = {"tsv_label_first", "fasttext"}

    if fmt not in allowed:
        raise ValueError(f"Unsupported train file format: {fmt}. Allowed: {allowed}")

    return path, fmt


def load_training_rows(train_file_specs: list[str]) -> list[dict[str, str]]:
    rows = []

    for spec in train_file_specs:
        path, fmt = parse_train_file_spec(spec)
        print(f"Loading training file: {path} [{fmt}]", flush=True)

        with path.open(encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, start=1):
                line = line.strip()

                if not line:
                    continue

                if fmt == "tsv_label_first":
                    parts = line.split("\t", 1)

                    if len(parts) != 2:
                        continue

                    label, text = parts

                elif fmt == "fasttext":
                    parts = line.split(" __label__", 1)

                    if len(parts) != 2:
                        continue

                    text = parts[0]
                    label = parts[1]

                else:
                    raise ValueError(f"Unsupported format: {fmt}")

                label_id = normalize_label(label)

                if not text or not label_id:
                    continue

                rows.append(
                    {
                        "text_a": text,
                        "label_id": label_id,
                        "source_file": str(path),
                        "line_no": str(line_no),
                    }
                )

    print(f"Loaded {len(rows)} training rows", flush=True)

    return rows


def load_dataset_language_filter(
    stats_path: str | Path | None,
    dataset_name: str | None,
) -> set[str] | None:
    """
    Optionally load the language set for a dataset from the original
    all_stats_pickle file.

    If stats_path or dataset_name is missing, no filtering is applied.
    """
    if stats_path is None or dataset_name is None:
        return None

    stats_path = require_file(stats_path)

    with stats_path.open("rb") as fr:
        stats = pickle.load(fr)

    if dataset_name not in stats:
        raise KeyError(
            f"Dataset {dataset_name!r} not found in stats file {stats_path}. "
            f"Available keys include: {list(stats.keys())[:20]}"
        )

    return {normalize_label(label) for label in stats[dataset_name]}


def stratified_sample_with_rare_all(
    train_rows: list[dict[str, Any]],
    n: int,
    rare_threshold: int,
    seed: int,
    label_key: str = "label_id",
    guarantee_nonrare: bool = True,
    allowed_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Include all items for rare labels and sample the remaining budget from
    non-rare labels.

    Rare labels are those with count < rare_threshold.

    If allowed_labels is provided, only those labels are considered.
    """
    rng = random.Random(seed)

    by_label = defaultdict(list)

    for row in train_rows:
        label = row[label_key]

        if allowed_labels is not None and label not in allowed_labels:
            continue

        by_label[label].append(row)

    if not by_label:
        raise ValueError("No rows left after label filtering.")

    rare_labels = {
        label
        for label, rows in by_label.items()
        if len(rows) < rare_threshold
    }

    nonrare_labels = [
        label
        for label in by_label.keys()
        if label not in rare_labels
    ]

    print(f"Rare labels: {len(rare_labels)}", flush=True)
    print(f"Non-rare labels: {len(nonrare_labels)}", flush=True)

    sampled = []

    for label in rare_labels:
        sampled.extend(by_label[label])

    if len(sampled) >= n:
        print(
            "Rare examples alone exceed target sample size; downsampling rare examples.",
            flush=True,
        )
        return rng.sample(sampled, n)

    remaining = n - len(sampled)
    remainder_pool = []

    if guarantee_nonrare:
        if remaining < len(nonrare_labels):
            raise ValueError(
                f"Not enough budget to guarantee one item per non-rare label. "
                f"remaining={remaining}, nonrare_labels={len(nonrare_labels)}. "
                f"Increase --sample-n or disable --guarantee-nonrare."
            )

        for label in nonrare_labels:
            rows = by_label[label]
            picked = rng.choice(rows)
            sampled.append(picked)

            rows_copy = rows.copy()
            rows_copy.remove(picked)
            remainder_pool.extend(rows_copy)
    else:
        for label in nonrare_labels:
            remainder_pool.extend(by_label[label])

    remaining = n - len(sampled)

    if remaining <= 0 or not remainder_pool:
        return sampled[:n]

    take = min(remaining, len(remainder_pool))
    sampled.extend(rng.sample(remainder_pool, take))

    print(f"Sampled rows: {len(sampled)}", flush=True)

    return sampled


def sample_with_median_temperature(
    data: list[dict[str, Any]],
    label_key: str = "label_id",
    alpha: float = 0.5,
    w_min: float = 0.1,
    w_max: float = 10.0,
    num_samples: int | None = None,
    replacement: bool = True,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Sample examples with median-centered temperature-like rebalancing.

    Label-level weight:
      weight(label) = (median_label_count / label_count) ** alpha

    Larger alpha gives stronger upsampling of low-frequency labels and
    stronger downsampling of high-frequency labels.
    """
    if not data:
        raise ValueError("Cannot sample from an empty dataset.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    labels = [example[label_key] for example in data]
    label_counts = Counter(labels)

    freqs = np.array(list(label_counts.values()), dtype=np.float64)
    median_count = float(np.median(freqs))

    label_weight = {
        label: (median_count / label_counts[label]) ** alpha
        for label in label_counts
    }

    weights = np.array([label_weight[label] for label in labels], dtype=np.float64)
    weights = np.clip(weights, w_min, w_max)

    weights_tensor = torch.as_tensor(weights, dtype=torch.double)

    if num_samples is None:
        num_samples = len(data)

    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=num_samples,
        replacement=replacement,
        generator=generator,
    )

    indices = list(sampler)

    return [data[i] for i in indices]


def plot_rank_frequency(
    before_counts: Counter,
    after_counts: Counter,
    output_path: str | Path,
    title: str = "Label distribution by rank",
    dpi: int = 300,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    labels = sorted(set(before_counts) | set(after_counts))

    before = np.array([before_counts.get(label, 0) for label in labels], dtype=float)
    after = np.array([after_counts.get(label, 0) for label in labels], dtype=float)

    order = np.argsort(-before)
    before_sorted = before[order]
    after_sorted = after[order]

    ranks = np.arange(1, len(labels) + 1)

    plt.figure()
    plt.plot(ranks, before_sorted, label="Before")
    plt.plot(ranks, after_sorted, label="After")
    plt.yscale("log")
    plt.xlabel("Label rank, sorted by original frequency")
    plt.ylabel("Count, log scale")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved distribution plot to {output_path}", flush=True)


def export_transformer_from_biencoder(
    biencoder_path: str | Path,
    export_dir: str | Path,
) -> str:
    """
    Export the underlying Hugging Face transformer and tokenizer from a
    SentenceTransformer bi-encoder checkpoint.

    The resulting directory can be used to initialize a CrossEncoder.
    """
    biencoder_path = require_file(biencoder_path)
    export_dir = ensure_dir(export_dir)

    print(f"Exporting transformer from bi-encoder: {biencoder_path}", flush=True)

    bi_encoder = SentenceTransformer(str(biencoder_path))
    transformer = bi_encoder[0]

    transformer.auto_model.save_pretrained(str(export_dir))
    transformer.tokenizer.save_pretrained(str(export_dir))

    print(f"Exported HF checkpoint to {export_dir}", flush=True)

    return str(export_dir)


@torch.no_grad()
def mine_hard_negatives(
    bi: SentenceTransformer,
    train_rows: list[dict[str, Any]],
    label_defs: dict[str, str],
    hard_k: int = 10,
    retrieve_k: int = 50,
    batch_size: int = 1000,
    temperature: float = 0.05,
    seed: int = 42,
) -> list[tuple[str, str, int]]:
    """
    Build binary cross-encoder training triples.

    Positive:
      (input_text, gold_label_description, 1)

    Negative:
      (input_text, hard_negative_label_description, 0)

    Hard negatives are sampled from the top retrieved label descriptions
    using a softmax over bi-encoder similarity scores.
    """
    rng = np.random.default_rng(seed)

    label_ids = list(label_defs.keys())
    label_texts = [label_defs[label_id] for label_id in label_ids]

    print("Precomputing label description embeddings...", flush=True)

    label_emb = bi.encode(
        label_texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    print("Encoding training texts...", flush=True)

    texts_a = [row["text_a"] for row in train_rows]

    text_emb = bi.encode(
        texts_a,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    triples = []

    print(f"Mining hard negatives for {len(train_rows)} rows...", flush=True)

    iterator = zip(train_rows, text_emb)

    for row, emb in tqdm.tqdm(iterator, total=len(train_rows)):
        gold = row["label_id"]

        if gold not in label_defs:
            continue

        scores = util.cos_sim(emb, label_emb).squeeze(0)

        top_indices = torch.topk(
            scores,
            k=min(retrieve_k, scores.shape[0]),
        ).indices.tolist()

        triples.append((row["text_a"], label_defs[gold], 1))

        candidate_label_ids = []
        candidate_scores = []

        for idx in top_indices:
            candidate_label = label_ids[idx]

            if candidate_label == gold:
                continue

            candidate_label_ids.append(candidate_label)
            candidate_scores.append(float(scores[idx]))

        if not candidate_label_ids:
            continue

        sample_size = min(hard_k, len(candidate_label_ids))

        candidate_scores = np.array(candidate_scores, dtype=np.float64)
        candidate_scores = candidate_scores / max(temperature, 1e-8)
        candidate_scores = candidate_scores - candidate_scores.max()

        probs = np.exp(candidate_scores)
        probs = probs / probs.sum()

        chosen_indices = rng.choice(
            np.arange(len(candidate_label_ids)),
            size=sample_size,
            replace=False,
            p=probs,
        )

        for idx in chosen_indices:
            negative_label = candidate_label_ids[idx]
            triples.append((row["text_a"], label_defs[negative_label], 0))

    print(f"Number of cross-encoder triples: {len(triples)}", flush=True)

    return triples


def train_cross_encoder_binary(
    hf_init_dir: str | Path,
    train_triples: list[tuple[str, str, int]],
    output_dir: str | Path,
    max_length: int = 256,
    batch_size: int = 48,
    epochs: int = 3,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
) -> None:
    """
    Train a cross-encoder as a binary classifier:
      (text, label_description) -> {0, 1}

    The model is initialized from a Hugging Face checkpoint exported from
    the bi-encoder transformer.
    """
    hf_init_dir = require_file(hf_init_dir)
    output_dir = ensure_dir(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Initializing CrossEncoder from {hf_init_dir} on device={device}",
        flush=True,
    )

    model = CrossEncoder(
        str(hf_init_dir),
        num_labels=1,
        max_length=max_length,
        device=device,
    )

    train_samples = [
        InputExample(texts=[text_a, label_desc], label=float(y))
        for text_a, label_desc, y in train_triples
    ]

    train_loader = DataLoader(
        train_samples,
        shuffle=True,
        batch_size=batch_size,
    )

    warmup_steps = math.ceil(len(train_loader) * epochs * warmup_ratio)

    print(
        f"Training cross-encoder: examples={len(train_samples)}, "
        f"batch_size={batch_size}, epochs={epochs}, lr={lr}, "
        f"warmup_steps={warmup_steps}, output={output_dir}",
        flush=True,
    )

    model.fit(
        train_dataloader=train_loader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=str(output_dir),
        show_progress_bar=True,
    )

    model.save(str(output_dir))

    print(f"Saved cross-encoder to {output_dir}", flush=True)


def validate_label_descriptions(
    train_rows: list[dict[str, Any]],
    language_descriptions: dict[str, str],
) -> dict[str, str]:
    labels = sorted({row["label_id"] for row in train_rows})

    missing = [label for label in labels if label not in language_descriptions]

    if missing:
        raise ValueError(
            f"{len(missing)} labels are missing from the language descriptions. "
            f"First missing labels: {missing[:20]}"
        )

    return {
        label: language_descriptions[label]
        for label in labels
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a language-description cross-encoder with hard negatives."
    )

    parser.add_argument(
        "--dataset-name",
        default="limit",
        help="Name used for logging and output file names.",
    )

    parser.add_argument(
        "--biencoder-path",
        required=True,
        help="Path to trained SentenceTransformer bi-encoder checkpoint.",
    )

    parser.add_argument(
        "--hf-init-dir",
        required=True,
        help="Directory where the exported HF transformer checkpoint will be written.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the trained cross-encoder will be saved.",
    )

    parser.add_argument(
        "--descriptions-path",
        default="./resources/WorldsLangs.tsv",
        help="Path to WorldsLangs.tsv or equivalent language-description TSV.",
    )

    parser.add_argument(
        "--train-file",
        action="append",
        required=True,
        help=(
            "Training file specification PATH:FORMAT. "
            "Supported formats: tsv_label_first, fasttext. "
            "May be provided multiple times."
        ),
    )

    parser.add_argument(
        "--stats-path",
        default=None,
        help="Optional path to all_stats_pickle for dataset-specific language filtering.",
    )

    parser.add_argument(
        "--filter-labels-with-stats",
        action="store_true",
        help="Use --stats-path and --dataset-name to filter labels.",
    )

    parser.add_argument("--sample-n", type=int, default=95_000)
    parser.add_argument("--rare-threshold", type=int, default=50)
    parser.add_argument("--disable-rare-sampling", action="store_true")
    parser.add_argument("--disable-guarantee-nonrare", action="store_true")

    parser.add_argument("--disable-median-temp-sampling", action="store_true")
    parser.add_argument("--sampler-alpha", type=float, default=0.85)
    parser.add_argument("--sampler-w-min", type=float, default=0.1)
    parser.add_argument("--sampler-w-max", type=float, default=8.0)

    parser.add_argument("--hard-k", type=int, default=5)
    parser.add_argument("--retrieve-k", type=int, default=20)
    parser.add_argument("--hard-negative-temperature", type=float, default=0.05)
    parser.add_argument("--encode-batch-size", type=int, default=1000)

    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=220)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--plot-path",
        default=None,
        help="Optional output path for rank-frequency plot, e.g. ./plots/dist_limit.png.",
    )

    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip exporting the transformer if --hf-init-dir already exists.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    configure_torch()

    language_descriptions = load_language_descriptions(args.descriptions_path)

    train_rows = load_training_rows(args.train_file)

    allowed_labels = None

    if args.filter_labels_with_stats:
        allowed_labels = load_dataset_language_filter(
            stats_path=args.stats_path,
            dataset_name=args.dataset_name,
        )
        print(
            f"Filtering to {len(allowed_labels)} labels from stats file",
            flush=True,
        )

    if not args.disable_rare_sampling:
        train_rows = stratified_sample_with_rare_all(
            train_rows=train_rows,
            n=args.sample_n,
            rare_threshold=args.rare_threshold,
            seed=args.seed,
            guarantee_nonrare=not args.disable_guarantee_nonrare,
            allowed_labels=allowed_labels,
        )
    elif allowed_labels is not None:
        train_rows = [
            row
            for row in train_rows
            if row["label_id"] in allowed_labels
        ]

    if not train_rows:
        raise ValueError("No training rows available after sampling/filtering.")

    lang_dist_before = Counter(row["label_id"] for row in train_rows)

    if not args.disable_median_temp_sampling:
        train_rows = sample_with_median_temperature(
            data=train_rows,
            alpha=args.sampler_alpha,
            w_min=args.sampler_w_min,
            w_max=args.sampler_w_max,
            num_samples=len(train_rows),
            seed=args.seed,
        )

    lang_dist_after = Counter(row["label_id"] for row in train_rows)

    print("Top label counts before/after sampling:", flush=True)
    for label, count_before in lang_dist_before.most_common(50):
        print(
            f"{label}: {count_before} -> {lang_dist_after[label]}",
            flush=True,
        )

    if args.plot_path:
        plot_rank_frequency(
            before_counts=lang_dist_before,
            after_counts=lang_dist_after,
            output_path=args.plot_path,
            title=f"Label distribution: {args.dataset_name}",
        )

    label_defs = validate_label_descriptions(
        train_rows=train_rows,
        language_descriptions=language_descriptions,
    )

    if args.skip_export:
        hf_init_dir = str(require_file(args.hf_init_dir))
    else:
        hf_init_dir = export_transformer_from_biencoder(
            biencoder_path=args.biencoder_path,
            export_dir=args.hf_init_dir,
        )

    bi_encoder = SentenceTransformer(args.biencoder_path)

    train_triples = mine_hard_negatives(
        bi=bi_encoder,
        train_rows=train_rows,
        label_defs=label_defs,
        hard_k=args.hard_k,
        retrieve_k=args.retrieve_k,
        batch_size=args.encode_batch_size,
        temperature=args.hard_negative_temperature,
        seed=args.seed,
    )

    train_cross_encoder_binary(
        hf_init_dir=hf_init_dir,
        train_triples=train_triples,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
    )


if __name__ == "__main__":
    main()