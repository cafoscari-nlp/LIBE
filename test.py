from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle as pkl
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# -----------------------------
# Data structures
# -----------------------------


@dataclass(frozen=True)
class Example:
    gold: str
    text: str


class CandidateMode(str, Enum):
    OPEN_DOMAIN = "open_domain"
    TRAIN_ONLY = "train_only"
    TRAIN_TEST_INTERSECTION = "train_test_intersection"


class DescriptionType(str, Enum):
    AUTO = "auto"
    FULL = "full"
    SUMMARIZED = "summarized"
    LABEL = "label"


@dataclass(frozen=True)
class Config:
    models: List[str]
    input_files: List[Path]
    out_dir: Path

    # Language-description resources
    label_desc_tsv: Path
    label_desc_summarized_tsv: Path
    description_type: DescriptionType

    # Optional training-label resources for restricted candidate spaces
    train_ds: Optional[str]
    datasets_stats_pkl: Optional[Path]

    # Input format: gold<TAB>text
    input_delimiter: str
    input_has_header: bool

    # Optional explicit candidate labels
    candidate_labels_tsv: Optional[Path]
    candidate_labels_list: Optional[List[str]]
    candidate_label_col: int
    candidate_delimiter: str
    candidate_has_header: bool

    # Candidate label space
    candidate_mode: CandidateMode

    # Retrieval and output
    top_k: int
    query_chunk_size: int
    output_prefix: str

    # Performance
    device: str
    batch_size: Optional[int]
    classifier_batch_size: Optional[int]
    desc_max_length: int
    query_max_length: int
    classifier_max_length: int
    desc_batch_size: Optional[int]
    query_batch_size: Optional[int]
    use_fp16: bool
    use_torch_compile: bool

    # Cache
    cache_dir: Optional[Path]


# -----------------------------
# Text / label utilities
# -----------------------------


_WS_RE = re.compile(r"\s+")
_CIT_RE = re.compile(r"\[\d+\]")


def normalize_label(label: str) -> str:
    """
    Normalize labels such as:
      __label__eng_Latn -> eng
      eng_Latn          -> eng
      eng               -> eng

    This keeps evaluation consistent with the training scripts.
    """
    label = str(label).strip()
    label = label.replace("__label__", "")
    return label.split("_")[0]


def clean_description(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\t", " ")
    text = _CIT_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def require_file(path: Path, description: str = "file") -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required {description} not found: {path}")
    return path


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()

    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested.startswith("cuda"):
        return requested if torch.cuda.is_available() else "cpu"

    if requested == "cpu":
        return "cpu"

    return "cpu"


def configure_torch_for_inference(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True


def read_rows(path: Path, delimiter: str) -> Iterable[List[str]]:
    require_file(path, "input file")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)

        for row in reader:
            if row:
                yield row


def sha1_str(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def model_display_name(model_path: str) -> str:
    return Path(model_path).name


def model_base_and_suffix(model_path: str) -> Tuple[str, str]:
    """
    Preserves the original script's '+0' suffix convention.

    If this convention is no longer needed, remove this function and use
    model_path directly.
    """
    if "+0" in str(model_path):
        return str(model_path).replace("+0", ""), "+0"

    return str(model_path), ""


# -----------------------------
# Data loading
# -----------------------------


def load_examples(
    path: Path,
    *,
    delimiter: str,
    has_header: bool,
) -> List[Example]:
    examples: List[Example] = []

    for row_idx, row in enumerate(read_rows(path, delimiter)):
        if has_header and row_idx == 0:
            continue

        if len(row) < 2:
            continue

        examples.append(
            Example(
                gold=normalize_label(row[0]),
                text=row[1],
            )
        )

    return examples


def collect_all_test_labels(
    input_files: Sequence[Path],
    delimiter: str,
    has_header: bool,
) -> List[str]:
    labels: List[str] = []

    for path in input_files:
        examples = load_examples(
            path,
            delimiter=delimiter,
            has_header=has_header,
        )
        labels.extend(example.gold for example in examples)

    return labels


def load_train_labels(cfg: Config) -> Optional[Set[str]]:
    """
    Load labels for the selected training dataset from a pickle.

    Supports both structures:
      stats[dataset] = dict-like
      stats[dataset] = list/set-like
    """
    if cfg.datasets_stats_pkl is None:
        return None

    require_file(cfg.datasets_stats_pkl, "dataset statistics pickle")

    if cfg.train_ds is None:
        raise ValueError("--train-ds is required when --datasets-stats-pkl is used.")

    with cfg.datasets_stats_pkl.open("rb") as f:
        ds_stats = pkl.load(f)

    if cfg.train_ds not in ds_stats:
        raise KeyError(
            f"Dataset {cfg.train_ds!r} not found in {cfg.datasets_stats_pkl}. "
            f"Available keys include: {list(ds_stats.keys())[:20]}"
        )

    items = ds_stats[cfg.train_ds]

    if isinstance(items, dict):
        labels = items.keys()
    else:
        labels = items

    return {normalize_label(label) for label in labels}


def load_candidate_labels(cfg: Config) -> Optional[List[str]]:
    """
    Load explicit candidate labels, either from --candidate-label or
    --candidate-labels-tsv.

    If neither is provided, returns None and candidate labels are determined
    from --candidate-mode.
    """
    labels: List[str] = []

    if cfg.candidate_labels_list is not None:
        labels.extend(cfg.candidate_labels_list)

    if cfg.candidate_labels_tsv is not None:
        require_file(cfg.candidate_labels_tsv, "candidate labels TSV")

        for row_idx, row in enumerate(
            read_rows(cfg.candidate_labels_tsv, cfg.candidate_delimiter)
        ):
            if cfg.candidate_has_header and row_idx == 0:
                continue

            if len(row) <= cfg.candidate_label_col:
                continue

            labels.append(row[cfg.candidate_label_col])

    if not labels:
        return None

    seen = set()
    out: List[str] = []

    for label in labels:
        label = normalize_label(label)

        if label not in seen:
            out.append(label)
            seen.add(label)

    return out


def load_label_descriptions(
    description_type: DescriptionType,
    *,
    full_path: Path,
    summarized_path: Path,
) -> Dict[str, str]:
    """
    Load label descriptions.

    description_type:
      full       -> full language descriptions
      summarized -> summarized language descriptions
      label      -> label string itself is used as the description
    """
    if description_type == DescriptionType.SUMMARIZED:
        path = summarized_path
    else:
        path = full_path

    require_file(path, "language-description TSV")

    print(f"Loading label descriptions from {path}", flush=True)

    descriptions: Dict[str, str] = {}

    with path.open("r", encoding="utf-8") as fr:
        reader = csv.reader(fr, delimiter="\t")

        for row in reader:
            if len(row) < 2:
                continue

            label = normalize_label(row[0])

            if description_type == DescriptionType.LABEL:
                descriptions[label] = label
            else:
                descriptions[label] = clean_description(row[1])

    return descriptions


def infer_description_type(model_path: str) -> DescriptionType:
    """
    Infer which label-description resource to use from the model path.

    This is kept for backward compatibility with your experiment naming.
    For final paper runs, prefer passing --description-type explicitly.
    """
    model_path_lower = model_path.lower()

    if "summarized" in model_path_lower or "summ" in model_path_lower:
        return DescriptionType.SUMMARIZED

    if "label" in model_path_lower or "classifier" in model_path_lower:
        return DescriptionType.LABEL

    if "full" in model_path_lower:
        return DescriptionType.FULL

    return DescriptionType.FULL


def build_candidate_labels(
    *,
    mode: CandidateMode,
    label_desc: Dict[str, str],
    test_labels: Sequence[str],
    train_labels: Optional[Set[str]],
    explicit_candidate_labels: Optional[Sequence[str]],
) -> List[str]:
    """
    Build the candidate label set used for evaluation.

    Priority:
      1. Explicit candidate labels, if provided.
      2. Candidate mode:
         - open_domain: all labels with descriptions
         - train_only: train labels from stats pickle
         - train_test_intersection: intersection of train and test labels
    """
    if explicit_candidate_labels is not None:
        base = [normalize_label(label) for label in explicit_candidate_labels]

    elif mode == CandidateMode.OPEN_DOMAIN:
        base = list(label_desc.keys())

    elif mode == CandidateMode.TRAIN_ONLY:
        if train_labels is None:
            raise ValueError(
                "--datasets-stats-pkl and --train-ds are required for "
                "--candidate-mode train_only unless explicit candidates are provided."
            )

        base = sorted(train_labels)

    elif mode == CandidateMode.TRAIN_TEST_INTERSECTION:
        if train_labels is None:
            raise ValueError(
                "--datasets-stats-pkl and --train-ds are required for "
                "--candidate-mode train_test_intersection unless explicit candidates are provided."
            )

        base = sorted(set(train_labels).intersection(test_labels))

    else:
        raise ValueError(f"Unknown candidate mode: {mode}")

    missing = [label for label in base if label not in label_desc]

    if missing:
        print(
            f"[WARN] Missing descriptions for {len(missing)} candidate labels. "
            f"First missing labels: {missing[:10]}",
            flush=True,
        )

    base = [label for label in base if label in label_desc]

    seen = set()
    deduped: List[str] = []

    for label in base:
        if label not in seen:
            deduped.append(label)
            seen.add(label)

    if not deduped:
        raise ValueError("Candidate label set is empty.")

    print(f"Candidate labels: {len(deduped)}", flush=True)

    return deduped


# -----------------------------
# Model detection
# -----------------------------


def is_classifier_model(model_path: str) -> bool:
    """
    Detect whether a model directory is a HF sequence-classification model.
    """
    base_model, _ = model_base_and_suffix(model_path)
    path = Path(base_model)
    config_path = path / "config.json"

    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            architectures = data.get("architectures", [])

            if any("SequenceClassification" in arch for arch in architectures):
                return True

            if "id2label" in data and "label2id" in data:
                return True

        except Exception:
            pass

    return False


# -----------------------------
# Encoding / caching
# -----------------------------


def autocast_context(device: str, use_fp16: bool):
    enabled = device.startswith("cuda") and use_fp16
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=enabled,
    )


def encode_texts_with_max_length(
    encoder: SentenceTransformer,
    texts: Sequence[str],
    *,
    device: str,
    batch_size: Optional[int],
    max_length: Optional[int],
    normalize: bool,
    use_fp16: bool,
) -> torch.Tensor:
    old_max_seq_length = getattr(encoder, "max_seq_length", None)

    if max_length is not None:
        encoder.max_seq_length = max_length

    try:
        with torch.inference_mode():
            with autocast_context(device, use_fp16):
                embeddings = encoder.encode(
                    texts,
                    convert_to_tensor=True,
                    device=device,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=normalize,
                )

        return embeddings

    finally:
        if old_max_seq_length is not None:
            encoder.max_seq_length = old_max_seq_length


def cache_key_for_labels(
    *,
    model_id: str,
    labels: Sequence[str],
    descriptions: Sequence[str],
    normalize: bool,
    desc_max_length: int,
) -> str:
    payload = json.dumps(
        {
            "model": model_id,
            "labels": list(labels),
            "descriptions_hash": sha1_str("\n".join(descriptions)),
            "normalize": normalize,
            "desc_max_length": desc_max_length,
        },
        sort_keys=True,
    )

    return sha1_str(payload)


def maybe_load_cached_embeddings(
    cache_dir: Path,
    key: str,
    device: str,
) -> Optional[torch.Tensor]:
    path = cache_dir / f"{key}.pt"

    if not path.exists():
        return None

    print(f"Loading cached label embeddings from {path}", flush=True)

    embeddings = torch.load(path, map_location="cpu")

    if device.startswith("cuda"):
        embeddings = embeddings.to(device, non_blocking=True)

    return embeddings


def maybe_save_cached_embeddings(
    cache_dir: Path,
    key: str,
    embeddings: torch.Tensor,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.pt"

    torch.save(embeddings.detach().cpu(), path)

    print(f"Saved label embeddings cache to {path}", flush=True)


# -----------------------------
# Retrieval model path
# -----------------------------


def build_retrieval_resources(
    cfg: Config,
    *,
    model_path: str,
    encoder: SentenceTransformer,
    label_desc: Dict[str, str],
    labels: Sequence[str],
) -> Tuple[List[str], torch.Tensor]:
    base_model, _ = model_base_and_suffix(model_path)
    device = resolve_device(cfg.device)

    labels = list(labels)
    descriptions = [label_desc[label] for label in labels]

    cache_key = cache_key_for_labels(
        model_id=base_model,
        labels=labels,
        descriptions=descriptions,
        normalize=True,
        desc_max_length=cfg.desc_max_length,
    )

    emb_labels: Optional[torch.Tensor] = None

    if cfg.cache_dir is not None:
        emb_labels = maybe_load_cached_embeddings(
            cfg.cache_dir,
            cache_key,
            device,
        )

    if emb_labels is None:
        print("Encoding label descriptions...", flush=True)

        emb_labels = encode_texts_with_max_length(
            encoder,
            descriptions,
            device=device,
            batch_size=cfg.desc_batch_size or cfg.batch_size,
            max_length=cfg.desc_max_length,
            normalize=True,
            use_fp16=cfg.use_fp16,
        )

        if cfg.cache_dir is not None:
            maybe_save_cached_embeddings(cfg.cache_dir, cache_key, emb_labels)

    return labels, emb_labels


def run_retrieval_model_on_file(
    cfg: Config,
    *,
    encoder: SentenceTransformer,
    model_path: str,
    input_path: Path,
    labels: Sequence[str],
    emb_labels: torch.Tensor,
) -> Path:
    base_model, suffix = model_base_and_suffix(model_path)
    model_name = model_display_name(base_model) + suffix
    device = resolve_device(cfg.device)

    examples = load_examples(
        input_path,
        delimiter=cfg.input_delimiter,
        has_header=cfg.input_has_header,
    )

    golds = [example.gold for example in examples]
    texts = [example.text for example in examples]

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    out_path = (
        cfg.out_dir
        / f"{cfg.output_prefix}_{input_path.stem}_{model_display_name(base_model)}{suffix}.tsv"
    )

    start = time.time()

    with out_path.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw, delimiter="\t")

        iterator = range(0, len(texts), cfg.query_chunk_size)

        for i in tqdm.tqdm(
            iterator,
            desc=f"{model_name}::{input_path.name}",
        ):
            batch_texts = texts[i : i + cfg.query_chunk_size]
            batch_golds = golds[i : i + cfg.query_chunk_size]

            emb_queries = encode_texts_with_max_length(
                encoder,
                batch_texts,
                device=device,
                batch_size=cfg.query_batch_size or cfg.batch_size,
                max_length=cfg.query_max_length,
                normalize=True,
                use_fp16=cfg.use_fp16,
            )

            hits_all = util.semantic_search(
                emb_queries,
                emb_labels,
                top_k=min(cfg.top_k, len(labels)),
                score_function=util.dot_score,
            )

            for gold, hits in zip(batch_golds, hits_all):
                row = [gold]

                for hit in hits:
                    label = labels[hit["corpus_id"]]
                    score = hit["score"]
                    row.append(f"{label}~{score}")

                writer.writerow(row)

    elapsed = time.time() - start

    print(
        f"[{model_name}] Retrieved {len(texts)} queries in {elapsed:.2f}s",
        flush=True,
    )
    print(f"Wrote: {out_path}", flush=True)

    return out_path


# -----------------------------
# Classifier model path
# -----------------------------


def load_classifier(
    base_model: str,
    device: str,
    use_torch_compile: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(base_model)
    model.eval().to(device)

    if device.startswith("cuda") and use_torch_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"[INFO] torch.compile enabled for {base_model}", flush=True)
        except Exception as exc:
            print(f"[WARN] torch.compile failed for {base_model}: {exc}", flush=True)

    return tokenizer, model


def classifier_id2label(model) -> Dict[int, str]:
    raw = model.config.id2label

    if not raw:
        raise ValueError("Classifier model has no id2label in its config.")

    return {
        int(index): normalize_label(label)
        for index, label in raw.items()
    }


def build_classifier_allowed_indices(
    *,
    id2label: Dict[int, str],
    candidate_labels: Sequence[str],
    device: str,
) -> Tuple[Optional[torch.Tensor], Dict[int, str]]:
    candidate_set = {normalize_label(label) for label in candidate_labels}

    allowed_original_ids = [
        idx
        for idx, label in sorted(id2label.items())
        if label in candidate_set
    ]

    if not allowed_original_ids:
        raise ValueError(
            "No overlap between classifier labels and candidate labels."
        )

    allowed_idx = torch.tensor(
        allowed_original_ids,
        device=device,
        dtype=torch.long,
    )

    restricted_id2label = {
        new_idx: id2label[old_idx]
        for new_idx, old_idx in enumerate(allowed_original_ids)
    }

    return allowed_idx, restricted_id2label


def run_classifier_model_on_file(
    cfg: Config,
    *,
    model_path: str,
    input_path: Path,
    tokenizer,
    model,
    candidate_labels: Sequence[str],
) -> Path:
    base_model, suffix = model_base_and_suffix(model_path)
    model_name = model_display_name(base_model) + suffix
    device = resolve_device(cfg.device)

    examples = load_examples(
        input_path,
        delimiter=cfg.input_delimiter,
        has_header=cfg.input_has_header,
    )

    golds = [example.gold for example in examples]
    texts = [example.text for example in examples]

    id2label = classifier_id2label(model)

    allowed_idx, current_id2label = build_classifier_allowed_indices(
        id2label=id2label,
        candidate_labels=candidate_labels,
        device=device,
    )

    batch_size = cfg.classifier_batch_size or cfg.batch_size or 512

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    out_path = (
        cfg.out_dir
        / f"{cfg.output_prefix}_{input_path.stem}_{model_display_name(base_model)}{suffix}.tsv"
    )

    start = time.time()

    with out_path.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw, delimiter="\t")

        with torch.inference_mode():
            iterator = range(0, len(texts), batch_size)

            for i in tqdm.tqdm(
                iterator,
                desc=f"{model_name}::{input_path.name}",
            ):
                batch_texts = texts[i : i + batch_size]
                batch_golds = golds[i : i + batch_size]

                encoded = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=cfg.classifier_max_length,
                    pad_to_multiple_of=8 if device.startswith("cuda") else None,
                )

                encoded = {
                    key: value.to(device, non_blocking=True)
                    for key, value in encoded.items()
                }

                with autocast_context(device, cfg.use_fp16):
                    logits = model(**encoded).logits

                logits = logits.index_select(dim=1, index=allowed_idx)

                k = min(cfg.top_k, logits.shape[1])
                top_logits, top_idx = torch.topk(logits, k=k, dim=-1)

                log_denom = torch.logsumexp(logits, dim=-1, keepdim=True)
                top_probs = torch.exp(top_logits - log_denom)

                top_idx_cpu = top_idx.cpu().tolist()
                top_probs_cpu = top_probs.float().cpu().tolist()

                for gold, idxs, probs in zip(
                    batch_golds,
                    top_idx_cpu,
                    top_probs_cpu,
                ):
                    row = [gold]

                    for idx, score in zip(idxs, probs):
                        row.append(f"{current_id2label[idx]}~{score}")

                    writer.writerow(row)

    elapsed = time.time() - start

    print(
        f"[{model_name}] Classified {len(texts)} queries in {elapsed:.2f}s",
        flush=True,
    )
    print(f"Wrote: {out_path}", flush=True)

    return out_path


# -----------------------------
# Core run
# -----------------------------


def description_type_for_model(
    cfg: Config,
    model_path: str,
) -> DescriptionType:
    if cfg.description_type != DescriptionType.AUTO:
        return cfg.description_type

    return infer_description_type(model_path)


def run_model_across_files(
    cfg: Config,
    *,
    model_path: str,
    input_files: Sequence[Path],
    label_desc: Dict[str, str],
    candidate_labels: Sequence[str],
) -> List[Path]:
    outputs: List[Path] = []
    base_model, _ = model_base_and_suffix(model_path)
    device = resolve_device(cfg.device)

    if is_classifier_model(model_path):
        print(f"[INFO] Detected classifier model: {model_path}", flush=True)

        tokenizer, model = load_classifier(
            base_model,
            device,
            cfg.use_torch_compile,
        )

        for input_path in input_files:
            outputs.append(
                run_classifier_model_on_file(
                    cfg,
                    model_path=model_path,
                    input_path=input_path,
                    tokenizer=tokenizer,
                    model=model,
                    candidate_labels=candidate_labels,
                )
            )

        return outputs

    print(f"[INFO] Detected retrieval model: {model_path}", flush=True)

    encoder = SentenceTransformer(base_model, device=device)

    labels, emb_labels = build_retrieval_resources(
        cfg,
        model_path=model_path,
        encoder=encoder,
        label_desc=label_desc,
        labels=candidate_labels,
    )

    for input_path in input_files:
        outputs.append(
            run_retrieval_model_on_file(
                cfg,
                encoder=encoder,
                model_path=model_path,
                input_path=input_path,
                labels=labels,
                emb_labels=emb_labels,
            )
        )

    return outputs


def run(cfg: Config) -> List[Path]:
    print("Evaluation started", flush=True)

    outputs: List[Path] = []

    train_labels = load_train_labels(cfg)
    explicit_candidate_labels = load_candidate_labels(cfg)

    all_test_labels = collect_all_test_labels(
        cfg.input_files,
        delimiter=cfg.input_delimiter,
        has_header=cfg.input_has_header,
    )

    desc_cache: Dict[DescriptionType, Dict[str, str]] = {}

    for model_path in cfg.models:
        print(f"[INFO] Loading model: {model_path}", flush=True)

        desc_type = description_type_for_model(cfg, model_path)

        if desc_type not in desc_cache:
            desc_cache[desc_type] = load_label_descriptions(
                desc_type,
                full_path=cfg.label_desc_tsv,
                summarized_path=cfg.label_desc_summarized_tsv,
            )

        label_desc = desc_cache[desc_type]

        candidate_labels = build_candidate_labels(
            mode=cfg.candidate_mode,
            label_desc=label_desc,
            test_labels=all_test_labels,
            train_labels=train_labels,
            explicit_candidate_labels=explicit_candidate_labels,
        )

        outputs.extend(
            run_model_across_files(
                cfg,
                model_path=model_path,
                input_files=cfg.input_files,
                label_desc=label_desc,
                candidate_labels=candidate_labels,
            )
        )

    print("Evaluation completed", flush=True)

    return outputs


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Run language-identification evaluation for SentenceTransformer "
            "retrieval models and Hugging Face classifier models. "
            "Input files must contain gold<TAB>text rows. "
            "Output rows contain gold followed by top-k label~score predictions."
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "One or more model directories or Hugging Face model IDs. "
            "Can mix SentenceTransformer retrieval models and classifier checkpoints."
        ),
    )

    parser.add_argument(
        "--input-files",
        nargs="+",
        type=Path,
        required=True,
        help="One or more TSV files with rows formatted as gold<TAB>text.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where prediction TSV files will be written.",
    )

    parser.add_argument(
        "--label-desc-tsv",
        type=Path,
        default=Path("./resources/WorldsLangs.tsv"),
        help="Full language-description TSV mapping label -> description.",
    )

    parser.add_argument(
        "--label-desc-summarized-tsv",
        type=Path,
        default=Path("./resources/WorldsLangs_Summarized.tsv"),
        help="Summarized language-description TSV mapping label -> description.",
    )

    parser.add_argument(
        "--description-type",
        choices=[item.value for item in DescriptionType],
        default=DescriptionType.AUTO.value,
        help=(
            "Which description text to use. "
            "auto infers from the model path; full uses --label-desc-tsv; "
            "summarized uses --label-desc-summarized-tsv; "
            "label uses the label string itself."
        ),
    )

    parser.add_argument(
        "--candidate-mode",
        choices=[mode.value for mode in CandidateMode],
        default=CandidateMode.TRAIN_ONLY.value,
        help=(
            "Candidate label space. "
            "open_domain = all labels with descriptions; "
            "train_only = labels from --datasets-stats-pkl and --train-ds; "
            "train_test_intersection = labels seen in both training and test."
        ),
    )

    parser.add_argument(
        "--train-ds",
        default=None,
        help=(
            "Training dataset key used with --datasets-stats-pkl. "
            "Required for train_only and train_test_intersection modes unless "
            "explicit candidate labels are supplied."
        ),
    )

    parser.add_argument(
        "--datasets-stats-pkl",
        type=Path,
        default=None,
        help=(
            "Optional pickle containing training dataset label statistics. "
            "Needed for train_only or train_test_intersection candidate modes."
        ),
    )

    parser.add_argument(
        "--candidate-labels-tsv",
        type=Path,
        default=None,
        help="Optional TSV file listing explicit candidate labels.",
    )

    parser.add_argument(
        "--candidate-label",
        action="append",
        default=None,
        help=(
            "Explicit candidate label. May be passed multiple times. "
            "If any explicit candidates are provided, they override --candidate-mode."
        ),
    )

    parser.add_argument("--candidate-label-col", type=int, default=0)
    parser.add_argument("--candidate-delimiter", default="\t")
    parser.add_argument("--candidate-has-header", action="store_true")

    parser.add_argument("--input-delimiter", default="\t")
    parser.add_argument("--input-has-header", action="store_true")

    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--query-chunk-size", type=int, default=10_000)
    parser.add_argument("--output-prefix", default="preds")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--classifier-batch-size", type=int, default=None)

    parser.add_argument("--desc-max-length", type=int, default=512)
    parser.add_argument("--query-max-length", type=int, default=96)
    parser.add_argument("--classifier-max-length", type=int, default=96)

    parser.add_argument("--desc-batch-size", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=None)

    parser.add_argument("--cache-dir", type=Path, default=None)

    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--no-use-fp16", dest="use_fp16", action="store_false")
    parser.set_defaults(use_fp16=True)

    parser.add_argument("--use-torch-compile", action="store_true")

    args = parser.parse_args()

    return Config(
        models=args.models,
        input_files=args.input_files,
        out_dir=args.out_dir,
        label_desc_tsv=args.label_desc_tsv,
        label_desc_summarized_tsv=args.label_desc_summarized_tsv,
        description_type=DescriptionType(args.description_type),
        train_ds=args.train_ds,
        datasets_stats_pkl=args.datasets_stats_pkl,
        input_delimiter=args.input_delimiter,
        input_has_header=args.input_has_header,
        candidate_labels_tsv=args.candidate_labels_tsv,
        candidate_labels_list=args.candidate_label,
        candidate_label_col=args.candidate_label_col,
        candidate_delimiter=args.candidate_delimiter,
        candidate_has_header=args.candidate_has_header,
        candidate_mode=CandidateMode(args.candidate_mode),
        top_k=args.top_k,
        query_chunk_size=args.query_chunk_size,
        output_prefix=args.output_prefix,
        device=args.device,
        batch_size=args.batch_size,
        classifier_batch_size=args.classifier_batch_size,
        desc_max_length=args.desc_max_length,
        query_max_length=args.query_max_length,
        classifier_max_length=args.classifier_max_length,
        desc_batch_size=args.desc_batch_size,
        query_batch_size=args.query_batch_size,
        use_fp16=args.use_fp16,
        use_torch_compile=args.use_torch_compile,
        cache_dir=args.cache_dir,
    )


def main() -> None:
    cfg = parse_args()

    device = resolve_device(cfg.device)

    print(
        f"Device resolved to: {device} "
        f"(cuda_available={torch.cuda.is_available()})",
        flush=True,
    )

    configure_torch_for_inference(device)

    run(cfg)


if __name__ == "__main__":
    main()