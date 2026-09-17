from __future__ import annotations
from eval2 import read_pred_tsv, normalize_label, choose_final_pred, compute_metrics,get_iso_langs,aggregate_group_metrics, FileResult, asdict, write_per_language_tsv
import argparse
import csv
import hashlib
import json
import pickle as pkl
import random
import re
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder, util


@dataclass(frozen=True)
class Example:
    gold: str
    text: str


class CandidateMode(str, Enum):
    OPEN_DOMAIN = "open_domain"
    TRAIN_ONLY = "train_only"
    TRAIN_TEST_INTERSECTION = "train_test_intersection"


@dataclass(frozen=True)
class Config:
    models: List[str]
    input_files: List[Path]
    train_ds: str
    out_dir: Path

    input_delimiter: str = "\t"
    input_has_header: bool = False

    candidate_labels_tsv: Optional[Path] = None
    datasets_stats_pkl: Optional[Path] = None
    candidate_label_col: int = 0
    candidate_delimiter: str = "\t"
    candidate_has_header: bool = False

    top_k: int = 5
    query_chunk_size: int = 100_000

    device: str = "cuda"
    batch_size: int = 8192
    desc_max_length: int = 64
    query_max_length: int = 64
    desc_batch_size: Optional[int] = None
    query_batch_size: Optional[int] = None
    use_fp16: bool = True

    cache_dir: Optional[Path] = None
    output_prefix: str = "preds"
    candidate_mode: CandidateMode = CandidateMode.TRAIN_ONLY

    limit_examples: Optional[int] = 1_000_000
    sample_with_replacement: bool = True

    run_fasttext: bool = True
    fasttext_path: Path = Path("../fastText")
    fasttext_model: Path = Path("../fastText/glotlid10M_new.bin")
    fasttext_batch_size: int = 100_000

    run_cross_encoder: bool = False
    cross_encoder_model: Optional[str] = None
    cross_encoder_threshold: float = 0.5
    cross_encoder_batch_size: int = 256
    cross_encoder_max_length: int = 512


_WS_RE = re.compile(r"\s+")
_CIT_RE = re.compile(r"\[\d+\]")


def clean_description(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\t", " ")
    text = _CIT_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda"):
        return requested if torch.cuda.is_available() else "cpu"
    return "cpu"


def configure_torch_for_inference(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True


def read_rows(path: Path, delimiter: str) -> Iterable[List[str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if row:
                yield row


def model_display_name(model_path: str) -> str:
    return Path(model_path).name


def model_base_and_suffix(model_path: str) -> Tuple[str, str]:
    if "+0" in str(model_path):
        return str(model_path).replace("+0", ""), "+0"
    return str(model_path), ""


def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def load_examples(
    path: Path,
    *,
    delimiter: str,
    has_header: bool,
    limit: Optional[int],
    sample_with_replacement: bool,
) -> List[Example]:
    examples: List[Example] = []
    print(f"[INFO] Loading examples from {path}...]")
    for row_idx, row in enumerate(read_rows(path, delimiter)):
        if has_header and row_idx == 0:
            continue
        if len(row) < 2:
            continue
        examples.append(Example(gold=row[0], text=row[1]))

    if limit is None:
        print(f"[INFO] {len(examples)} examples loaded. ")
        return examples

    print(f"[INFO] {limit} examples loaded. ")
    if sample_with_replacement:
        return random.choices(examples, k=limit)

    return examples[:limit]


def load_train_labels(cfg: Config) -> Optional[Set[str]]:
    if cfg.datasets_stats_pkl is None:
        return None

    with cfg.datasets_stats_pkl.open("rb") as f:
        ds_stats = pkl.load(f)

    return {d.split("_")[0] for d in ds_stats[cfg.train_ds].keys() if d!=""}


def load_candidate_labels(cfg: Config) -> Optional[List[str]]:
    if cfg.candidate_labels_tsv is None:
        return None

    labels: List[str] = []

    for row_idx, row in enumerate(read_rows(cfg.candidate_labels_tsv, cfg.candidate_delimiter)):
        if cfg.candidate_has_header and row_idx == 0:
            continue
        if len(row) <= cfg.candidate_label_col:
            continue
        labels.append(row[cfg.candidate_label_col])

    seen = set()
    out = []

    for label in labels:
        if label not in seen:
            out.append(label)
            seen.add(label)

    return out


def load_label_descriptions(description: str) -> Dict[str, str]:
    print("Loading language descriptions...")

    if "summ" in description:
        fname = "./resources/WorldsLangs_Summarized.tsv"
    else:
        fname = "./resources/WorldsLangs.tsv"

    langs: Dict[str, str] = {}

    with open(fname, "r", encoding="utf-8") as fr:
        reader = csv.reader(fr, delimiter="\t")

        for row in reader:
            if len(row) < 2:
                continue

            if description == "label":
                langs[row[0]] = row[0]
            else:
                langs[row[0]] = clean_description(row[1])

    return langs


def get_description_type(model_path: str) -> str:
    mp = model_path.lower()

    if "summarized" in mp:
        return "summ"
    if "full" in mp:
        return "full"
    if "label" in mp or "classifier" in mp:
        return "label"

    return "full"


def build_candidate_labels(
    *,
    mode: CandidateMode,
    label_desc: Dict[str, str],
    test_labels: Sequence[str],
    train_labels: Optional[Set[str]],
    candidate_labels: Optional[Sequence[str]],
) -> List[str]:
    if candidate_labels is not None:
        base = list(candidate_labels)
    elif mode == CandidateMode.OPEN_DOMAIN:
        base = list(label_desc.keys())
    elif mode == CandidateMode.TRAIN_ONLY:
        if train_labels is None:
            raise ValueError("train_labels is required for mode=train_only")
        base = sorted(train_labels)
    elif mode == CandidateMode.TRAIN_TEST_INTERSECTION:
        if train_labels is None:
            raise ValueError("train_labels is required for mode=train_test_intersection")
        base = sorted(set(train_labels).intersection(test_labels))
    else:
        raise ValueError(f"Unknown candidate mode: {mode}")

    missing = [label for label in base if label not in label_desc]

    if missing:
        print(f"[WARN] Missing descriptions for {len(missing)} labels. First 10: {missing[:10]}")

    base = [label for label in base if label in label_desc]

    seen = set()
    out = []

    for label in base:
        if label not in seen:
            out.append(label)
            seen.add(label)

    return out


def autocast_context(device: str, use_fp16: bool):
    enabled = device.startswith("cuda") and use_fp16
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)


def cache_key_for_labels(
    *,
    model_id: str,
    labels: Sequence[str],
    max_length: int,
    normalize: bool,
) -> str:
    payload = json.dumps(
        {
            "model": model_id,
            "labels": list(labels),
            "max_length": max_length,
            "normalize": normalize,
        },
        sort_keys=True,
    )
    return sha1_str(payload)


def maybe_load_cached_embeddings(cache_dir: Path, key: str, device: str) -> Optional[torch.Tensor]:
    path = cache_dir / f"{key}.pt"

    if not path.exists():
        return None

    emb = torch.load(path, map_location="cpu")

    if device.startswith("cuda"):
        emb = emb.to(device, non_blocking=True)

    return emb


def maybe_save_cached_embeddings(cache_dir: Path, key: str, emb: torch.Tensor) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.pt"
    torch.save(emb.detach().cpu(), path)


@torch.inference_mode()
def encode_texts_with_max_length(
    encoder: SentenceTransformer,
    texts: Sequence[str],
    *,
    device: str,
    batch_size: int,
    max_length: int,
    normalize: bool,
    use_fp16: bool,
) -> torch.Tensor:
    old_max_seq_length = getattr(encoder, "max_seq_length", None)
    encoder.max_seq_length = max_length

    try:
        with autocast_context(device, use_fp16):
            emb = encoder.encode(
                list(texts),
                convert_to_tensor=True,
                device=device,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize,
            )
        return emb
    finally:
        if old_max_seq_length is not None:
            encoder.max_seq_length = old_max_seq_length


def pretokenize_texts_with_max_length(
    encoder: SentenceTransformer,
    texts: Sequence[str],
    *,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    #old_max_seq_length = getattr(encoder, "max_seq_length", None)
    encoder.max_seq_length = max_length

    #try:
    return encoder.tokenize(list(texts))
    #finally:
    #    if old_max_seq_length is not None:
    #        encoder.max_seq_length = old_max_seq_length


def slice_features(
    features: Dict[str, torch.Tensor],
    start: int,
    end: int,
) -> Dict[str, torch.Tensor]:
    return {
        key: value[start:end] if torch.is_tensor(value) else value
        for key, value in features.items()
    }


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: str,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.inference_mode()
def encode_pretokenized_timed(
    encoder: SentenceTransformer,
    features: Dict[str, torch.Tensor],
    *,
    device: str,
    batch_size: int,
    normalize: bool,
    use_fp16: bool,
) -> Tuple[torch.Tensor, float]:
    encoder.eval()
    input_ids = features["input_ids"]
    n = input_ids.size(0)
    output_embs: Optional[torch.Tensor] = None
    write_pos = 0
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    features = {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in features.items()
    }
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        #batch = {k: v[i:end] for k, v in features.items()}
        batch = {k: v[i:end] for k, v in features.items() if k != "modality"}

        with autocast_context(device, use_fp16):
            model = encoder[0].auto_model
            outputs = model(**batch)
            token_emb = outputs.last_hidden_state
            attention_mask = batch["attention_mask"]
            mask = attention_mask.unsqueeze(-1).float()
            emb = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            if normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        if output_embs is None:
            output_embs = torch.empty(
                (n, emb.shape[1]),
                dtype=emb.dtype,
                device=emb.device,
            )

        output_embs[write_pos:write_pos + emb.size(0)].copy_(emb)
        write_pos += emb.size(0)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    assert output_embs is not None
    return output_embs, elapsed


def collect_all_test_labels(
    input_files: Sequence[Path],
    *,
    delimiter: str,
    has_header: bool,
    limit: Optional[int],
    sample_with_replacement: bool,
) -> List[str]:
    labels: List[str] = []

    for path in input_files:
        examples = load_examples(
            path,
            delimiter=delimiter,
            has_header=has_header,
            limit=limit,
            sample_with_replacement=sample_with_replacement,
        )
        labels.extend(ex.gold for ex in examples)

    return labels


def build_retrieval_resources(
    cfg: Config,
    *,
    model_path: str,
    encoder: SentenceTransformer,
    label_desc: Dict[str, str],
    candidate_labels: Optional[Sequence[str]],
    train_labels: Optional[Set[str]],
    all_test_labels: Sequence[str],
) -> Tuple[List[str], torch.Tensor]:
    base_model, _ = model_base_and_suffix(model_path)
    device = resolve_device(cfg.device)

    labels = build_candidate_labels(
        mode=cfg.candidate_mode,
        label_desc=label_desc,
        test_labels=all_test_labels,
        train_labels=train_labels,
        candidate_labels=candidate_labels,
    )

    cache_key = cache_key_for_labels(
        model_id=base_model,
        labels=labels,
        max_length=cfg.desc_max_length,
        normalize=True,
    )

    emb_labels = None

    if cfg.cache_dir is not None:
        emb_labels = maybe_load_cached_embeddings(cfg.cache_dir, cache_key, device)

    if emb_labels is None:
        descriptions = [label_desc[label] for label in labels]

        start = time.perf_counter()

        emb_labels = encode_texts_with_max_length(
            encoder,
            descriptions,
            device=device,
            batch_size=cfg.desc_batch_size or cfg.batch_size,
            max_length=cfg.desc_max_length,
            normalize=True,
            use_fp16=cfg.use_fp16,
        )

        if device.startswith("cuda"):
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        print(
            f"[DESC] Embedded {len(descriptions):,} descriptions "
            f"in {elapsed:.2f}s "
            f"({len(descriptions) / elapsed:.1f} desc/s)"
        )

        if cfg.cache_dir is not None:
            maybe_save_cached_embeddings(cfg.cache_dir, cache_key, emb_labels)

    return labels, emb_labels

def compute_pretokenized_length_stats(
    features: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    if "attention_mask" in features:
        lengths = features["attention_mask"].sum(dim=1).cpu().numpy()
    else:
        pad_id = 0
        lengths = (features["input_ids"] != pad_id).sum(dim=1).cpu().numpy()

    stats = {
        "mean": float(lengths.mean()),
        "p50": float(np.percentile(lengths, 50)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
    }

    print("\n[Pre-tokenized Length Stats]")
    for k, v in stats.items():
        print(f"{k}: {v}")

    return stats

def eval_file(pred_path: Path,
              thr=.0,
              reject_label = "__REJECT__",
              threshold_inclusive = True,
              realistic = False,
              per_language_tsv_dir = Path("./results/per_language_metrics")) -> Dict[str, float]:
    results: List[FileResult] = []
    if not pred_path.exists():
        print(f"[WARN] prediction file not found: {pred_path}; skipping.")
        return {}

    golds, all_candidates = read_pred_tsv(pred_path)
    gold_set = set([normalize_label(g) for g in golds])

    y_true = []
    y_pred = []
    for gold, candidates in zip(golds, all_candidates):
        final = choose_final_pred(
                    candidates,
                    threshold=thr,
                    reject_label=reject_label,
                    inclusive=threshold_inclusive,
                    realistic=realistic,
                    gold_set=gold_set,
                )
        y_true.append(normalize_label(gold))
        y_pred.append(normalize_label(final))

    # basic metrics
    labels, macro, per_class = compute_metrics(y_true, y_pred, reject_label)

    # seen / unseen LaBSE aggregation
    iso_langs = get_iso_langs(str(pred_path))
    seen_labels = {pc.label for pc in per_class if pc.label in iso_langs}
    unseen_labels = {pc.label for pc in per_class if pc.label != reject_label and pc.label not in iso_langs}

    seen_group = aggregate_group_metrics(
                per_class,
                labels_in_group=seen_labels,
                reject_label=reject_label,
                name="seen_in_labse",
            )
    unseen_group = aggregate_group_metrics(
                per_class,
                labels_in_group=unseen_labels,
                reject_label=reject_label,
                name="unseen_in_labse",
            )

            # coverage + accuracy on accepted
    accepted_mask = [p != reject_label for p in y_pred]
    num_accepted = sum(accepted_mask)
    coverage = num_accepted / len(y_pred) if y_pred else 0.0
    num_correct_accepted = sum(
                1 for gt, p in zip(y_true, y_pred)
                if p != reject_label and p == gt
            )
    acc_on_accepted = (num_correct_accepted / num_accepted) if num_accepted else 0.0

    fr = FileResult(
                pred_file=str(pred_path),
                threshold=float(thr),
                realistic=realistic,
                labels=labels,
                macro=macro,
                per_class=[asdict(pc) for pc in per_class],
                coverage=float(coverage),
                accepted=int(num_accepted),
                acc_on_accepted=float(acc_on_accepted),
                seen_macro={
                    "precision": seen_group.macro_precision,
                    "recall": seen_group.macro_recall,
                    "f1": seen_group.macro_f1,
                    "fpr": seen_group.macro_fpr,
                    "num_labels": seen_group.num_labels,
                },
                unseen_macro={
                    "precision": unseen_group.macro_precision,
                    "recall": unseen_group.macro_recall,
                    "f1": unseen_group.macro_f1,
                    "fpr": unseen_group.macro_fpr,
                    "num_labels": unseen_group.num_labels,
                },
            )
    results.append(fr)

    # optional per-language TSV
    if per_language_tsv_dir is not None:
        out_name = f"{pred_path.stem}__thr_{str(thr).replace('.', 'p')}__per_language.tsv"
        write_per_language_tsv(
            per_language_tsv_dir / out_name,
            pred_file=str(pred_path),
            threshold=thr,
            realistic=realistic,
            per_class=per_class,
            reject_label=reject_label,
        )

    # Print summary
    file_stem = pred_path.stem
    print(
                f"{file_stem}\trealistic={realistic}\tT={thr}\n"
                f"P={macro['precision']:.3f}\tR={macro['recall']:.3f}\t"
                f"F1={macro['f1']:.3f}\tFPR={macro['fpr']:.4f}\n"
                f"coverage={coverage:.3f}\taccepted={num_accepted}/{len(y_pred)}\t"
                f"acc_on_accepted={acc_on_accepted:.3f}\n"
                f"SEEN_IN_LABSE\tn={seen_group.num_labels}\t"
                f"P={seen_group.macro_precision:.3f}\tR={seen_group.macro_recall:.3f}\t"
                f"F1={seen_group.macro_f1:.3f}\tFPR={seen_group.macro_fpr:.4f}\n"
                f"UNSEEN_IN_LABSE\tn={unseen_group.num_labels}\t"
                f"P={unseen_group.macro_precision:.3f}\tR={unseen_group.macro_recall:.3f}\t"
                f"F1={unseen_group.macro_f1:.3f}\tFPR={unseen_group.macro_fpr:.4f}\n"
            )

    return results

def rerank_low_confidence_with_cross_encoder(
    *,
    cross_encoder: CrossEncoder,
    batch_texts: Sequence[str],
    hits_all: Sequence[List[Dict]],
    labels: Sequence[str],
    label_desc: Dict[str, str],
    threshold: float,
    batch_size: int,
    device: str,
) -> Tuple[List[List[Dict]], float, int, int]:
    """
    Collect all samples whose bi-encoder top-1 score is below threshold,
    then activate the cross-encoder only on those collected samples.

    The cross-encoder reranks the existing bi-encoder top-k candidates.
    """
    reranked_hits_all: List[List[Dict]] = [
        [dict(h) for h in hits]
        for hits in hits_all
    ]

    low_conf_indices: List[int] = []

    for local_i, hits in enumerate(hits_all):
        if not hits:
            continue

        bi_top1_score = float(hits[0]["score"])

        if bi_top1_score < threshold:
            low_conf_indices.append(local_i)

    if not low_conf_indices:
        return reranked_hits_all, 0.0, 0, 0

    pairs: List[Tuple[str, str]] = []
    pair_metadata: List[Tuple[int, int]] = []

    for local_i in low_conf_indices:
        query_text = batch_texts[local_i]

        for hit_pos, hit in enumerate(hits_all[local_i]):
            label_id = int(hit["corpus_id"])
            label = labels[label_id]
            description = label_desc[label]

            pairs.append((query_text, description))
            pair_metadata.append((local_i, hit_pos))

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.perf_counter()

    cross_scores = cross_encoder.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    cross_time = time.perf_counter() - start

    scores_by_query: Dict[int, List[Tuple[int, float]]] = {}

    for score, (local_i, hit_pos) in zip(cross_scores, pair_metadata):
        scores_by_query.setdefault(local_i, []).append((hit_pos, float(score)))

    for local_i, hit_scores in scores_by_query.items():
        original_hits = reranked_hits_all[local_i]

        reranked = sorted(
            hit_scores,
            key=lambda x: x[1],
            reverse=True,
        )

        new_hits: List[Dict] = []

        for hit_pos, ce_score in reranked:
            h = dict(original_hits[hit_pos])

            h["bi_score"] = float(h["score"])
            h["score"] = float(ce_score)
            h["score_source"] = "cross_encoder"

            new_hits.append(h)

        reranked_hits_all[local_i] = new_hits

    return reranked_hits_all, cross_time, len(low_conf_indices), len(pairs)

def compute_simple_prf(
    *,
    golds: Sequence[str],
    preds: Sequence[str],
    reject_label: str = "__REJECT__",
) -> Dict[str, float]:
    """
    Computes macro precision, recall, and F1 using the same eval2 utilities.
    Labels are normalized before scoring.
    """
    if not golds:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    y_true = [normalize_label(g) for g in golds]
    y_pred = [normalize_label(p) for p in preds]

    _, macro, _ = compute_metrics(y_true, y_pred, reject_label)

    return {
        "precision": float(macro["precision"]),
        "recall": float(macro["recall"]),
        "f1": float(macro["f1"]),
    }

def run_retrieval_model_on_file(
    cfg: Config,
    *,
    encoder: SentenceTransformer,
    cross_encoder: Optional[CrossEncoder],
    model_path: str,
    input_path: Path,
    labels: Sequence[str],
    label_desc: Dict[str, str],
    emb_labels: torch.Tensor,
) -> Path:
    base_model, suffix = model_base_and_suffix(model_path)
    model_name = model_display_name(base_model) + suffix
    device = resolve_device(cfg.device)

    examples = load_examples(
        input_path,
        delimiter=cfg.input_delimiter,
        has_header=cfg.input_has_header,
        limit=cfg.limit_examples,
        sample_with_replacement=cfg.sample_with_replacement,
    )

    golds = [ex.gold for ex in examples]
    texts = [ex.text for ex in examples]
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    timing_path = cfg.out_dir / (
        f"{cfg.output_prefix}_{input_path.stem}_{model_display_name(base_model)}"
        f"{suffix}_timings.tsv"
    )

    details_path = cfg.out_dir / (
        f"{cfg.output_prefix}_{input_path.stem}_{model_display_name(base_model)}"
        f"{suffix}_score_details.tsv"
    )

    last_out_path: Optional[Path] = None

    with timing_path.open("w", encoding="utf-8", newline="") as tfw, \
         details_path.open("w", encoding="utf-8", newline="") as dfw:

        timing_writer = csv.writer(tfw, delimiter="\t")
        details_writer = csv.writer(dfw, delimiter="\t")

        timing_writer.writerow(
            [
                "input_file",
                "model",
                "max_seq_len",
                "examples",
                "top_k",

                "tokenization_time",
                "embedding_time",
                "similarity_time",
                "cross_encoder_time",
                "e2e_excluding_tokenization",
                "wall_time",

                "tokenization_throughput",
                "embedding_throughput",
                "similarity_throughput",
                "cross_encoder_query_throughput",
                "cross_encoder_pair_throughput",
                "e2e_excluding_tokenization_throughput",
                "e2e_including_tokenization_throughput",

                "rerouted_queries",
                "rerouted_fraction",
                "cross_encoder_pairs",

                "cross_encoder_precision",
                "cross_encoder_recall",
                "cross_encoder_f1",

                "final_precision",
                "final_recall",
                "final_f1",

                "cross_encoder_enabled",
                "cross_encoder_model",
                "cross_encoder_threshold",

                "peak_allocated_mb",
                "peak_reserved_mb",
            ]
        )

        details_writer.writerow(
            [
                "max_seq_len",
                "row_index",
                "gold",
                "final_rank",
                "label",
                "final_score",
                "score_source",
                "bi_score",
            ]
        )

        for max_seq_len in [16, 32, 64, 96, 112]:
            out_path = cfg.out_dir / (
                f"{cfg.output_prefix}_{input_path.stem}_{model_display_name(base_model)}"
                f"{suffix}_seq{max_seq_len}.tsv"
            )
            last_out_path = out_path

            with out_path.open("w", encoding="utf-8", newline="") as fw:
                writer = csv.writer(fw, delimiter="\t")

                total_embed_only_time = 0.0
                total_similarity_time = 0.0
                total_tokenization_time = 0.0
                total_cross_encoder_time = 0.0

                total_rerouted_queries = 0
                total_cross_encoder_pairs = 0

                final_metric_golds: List[str] = []
                final_metric_preds: List[str] = []

                cross_metric_golds: List[str] = []
                cross_metric_preds: List[str] = []

                wall_start = time.perf_counter()

                print("\n-------------------------\n")
                print(f"max_seq_len: {max_seq_len}")

                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                global_row_index = 0

                for i in tqdm.tqdm(
                    range(0, len(texts), cfg.query_chunk_size),
                    desc=f"{model_name}::{input_path.name}::seq{max_seq_len}",
                ):
                    batch_texts = texts[i:i + cfg.query_chunk_size]
                    batch_golds = golds[i:i + cfg.query_chunk_size]

                    tok_start = time.perf_counter()

                    tokenized_queries = pretokenize_texts_with_max_length(
                        encoder,
                        batch_texts,
                        max_length=max_seq_len,
                    )

                    if device.startswith("cuda"):
                        tokenized_queries = {
                            k: v.pin_memory() if torch.is_tensor(v) else v
                            for k, v in tokenized_queries.items()
                        }

                    total_tokenization_time += time.perf_counter() - tok_start

                    if max_seq_len > 100:
                        temp_batch_size = 1_000 * 10
                    elif max_seq_len > 90:
                        temp_batch_size = 1_200 * 10
                    elif max_seq_len > 60:
                        temp_batch_size = 2_000 * 10
                    elif max_seq_len > 30:
                        temp_batch_size = 3_000 * 10
                    else:
                        temp_batch_size = 8_000 * 10

                    emb_queries, embed_only_time = encode_pretokenized_timed(
                        encoder,
                        tokenized_queries,
                        device=device,
                        batch_size=temp_batch_size,
                        normalize=True,
                        use_fp16=cfg.use_fp16,
                    )

                    total_embed_only_time += embed_only_time

                    if device.startswith("cuda"):
                        torch.cuda.synchronize()

                    sim_start = time.perf_counter()

                    hits_all = util.semantic_search(
                        emb_queries,
                        emb_labels,
                        top_k=cfg.top_k,
                        query_chunk_size=min(len(emb_queries), 8192),
                        corpus_chunk_size=len(emb_labels),
                        score_function=util.dot_score,
                    )

                    if device.startswith("cuda"):
                        torch.cuda.synchronize()

                    similarity_time = time.perf_counter() - sim_start
                    total_similarity_time += similarity_time

                    cross_time = 0.0
                    rerouted_queries = 0
                    cross_encoder_pairs = 0

                    if cfg.run_cross_encoder and cross_encoder is not None:
                        hits_all, cross_time, rerouted_queries, cross_encoder_pairs = (
                            rerank_low_confidence_with_cross_encoder(
                                cross_encoder=cross_encoder,
                                batch_texts=batch_texts,
                                hits_all=hits_all,
                                labels=labels,
                                label_desc=label_desc,
                                threshold=cfg.cross_encoder_threshold,
                                batch_size=cfg.cross_encoder_batch_size,
                                device=device,
                            )
                        )

                        total_cross_encoder_time += cross_time
                        total_rerouted_queries += rerouted_queries
                        total_cross_encoder_pairs += cross_encoder_pairs

                    for local_row_index, (gold, hits) in enumerate(zip(batch_golds, hits_all)):
                        row = [gold] + [
                            f"{labels[int(h['corpus_id'])]}~{float(h['score'])}"
                            for h in hits
                        ]
                        writer.writerow(row)

                        if hits:
                            final_top1_label = labels[int(hits[0]["corpus_id"])]
                            final_metric_golds.append(gold)
                            final_metric_preds.append(final_top1_label)

                            if hits[0].get("score_source", "bi_encoder") == "cross_encoder":
                                cross_metric_golds.append(gold)
                                cross_metric_preds.append(final_top1_label)

                        for rank, h in enumerate(hits, start=1):
                            label = labels[int(h["corpus_id"])]
                            final_score = float(h["score"])
                            score_source = h.get("score_source", "bi_encoder")
                            bi_score = h.get("bi_score", final_score)

                            details_writer.writerow(
                                [
                                    max_seq_len,
                                    global_row_index + local_row_index,
                                    gold,
                                    rank,
                                    label,
                                    f"{final_score:.8f}",
                                    score_source,
                                    f"{float(bi_score):.8f}",
                                ]
                            )

                    global_row_index += len(batch_texts)

                    if embed_only_time > 0:
                        print(
                            f"[LABSE CHUNK] {len(batch_texts):,} texts | "
                            f"embed_only={embed_only_time:.2f}s | "
                            f"embed_throughput={len(batch_texts) / embed_only_time:.1f} texts/s | "
                            f"similarity={similarity_time:.2f}s | "
                            f"cross={cross_time:.2f}s | "
                            f"rerouted={rerouted_queries:,} | "
                            f"cross_pairs={cross_encoder_pairs:,}"
                        )

                wall_time = time.perf_counter() - wall_start
                n = len(texts)

                e2e_excl_tokenization_time = (
                    total_embed_only_time
                    + total_similarity_time
                    + total_cross_encoder_time
                )

                peak_allocated_mb = 0.0
                peak_reserved_mb = 0.0

                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                    peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
                    peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024 ** 2

                tokenization_thr = (
                    n / total_tokenization_time
                    if total_tokenization_time > 0
                    else 0.0
                )
                embedding_thr = (
                    n / total_embed_only_time
                    if total_embed_only_time > 0
                    else 0.0
                )
                similarity_thr = (
                    n / total_similarity_time
                    if total_similarity_time > 0
                    else 0.0
                )
                cross_query_thr = (
                    total_rerouted_queries / total_cross_encoder_time
                    if total_cross_encoder_time > 0
                    else 0.0
                )
                cross_pair_thr = (
                    total_cross_encoder_pairs / total_cross_encoder_time
                    if total_cross_encoder_time > 0
                    else 0.0
                )
                e2e_excl_tok_thr = (
                    n / e2e_excl_tokenization_time
                    if e2e_excl_tokenization_time > 0
                    else 0.0
                )
                wall_thr = n / wall_time if wall_time > 0 else 0.0
                rerouted_fraction = total_rerouted_queries / n if n else 0.0

                print()
                print(f"[LABSE] Finished {input_path}")
                print(f"max_seq_len: {max_seq_len}")
                print(f"Examples: {n:,}")
                print(f"Wall time: {wall_time:.2f}s")
                print(f"Tokenization time, excluded from E2E: {total_tokenization_time:.2f}s")
                print(f"End-to-end time excluding tokenization: {e2e_excl_tokenization_time:.2f}s")
                print(f"Embedding-only time: {total_embed_only_time:.2f}s")
                print(f"Similarity time: {total_similarity_time:.2f}s")
                print(f"Cross-encoder time: {total_cross_encoder_time:.2f}s")
                print(f"Rerouted queries: {total_rerouted_queries:,}/{n:,}")
                print(f"Rerouted fraction: {rerouted_fraction:.6f}")
                print(f"Cross-encoder pairs: {total_cross_encoder_pairs:,}")

                if device.startswith("cuda"):
                    print(f"Peak allocated: {peak_allocated_mb:.2f} MB")
                    print(f"Peak reserved:  {peak_reserved_mb:.2f} MB")

                print(f"Tokenization throughput: {tokenization_thr:.1f} texts/s")
                print(f"Embedding-only throughput: {embedding_thr:.1f} texts/s")
                print(f"Similarity throughput: {similarity_thr:.1f} texts/s")
                print(f"Cross-encoder query throughput: {cross_query_thr:.1f} queries/s")
                print(f"Cross-encoder pair throughput: {cross_pair_thr:.1f} pairs/s")
                print(f"End-to-end throughput excluding tokenization: {e2e_excl_tok_thr:.1f} texts/s")
                print(f"End-to-end throughput including tokenization: {wall_thr:.1f} texts/s")

                cross_metrics = compute_simple_prf(
                    golds=cross_metric_golds,
                    preds=cross_metric_preds,
                )

                final_metrics = compute_simple_prf(
                    golds=final_metric_golds,
                    preds=final_metric_preds,
                )

                print(
                    f"Cross-encoder subset P/R/F1: "
                    f"{cross_metrics['precision']:.3f} / "
                    f"{cross_metrics['recall']:.3f} / "
                    f"{cross_metrics['f1']:.3f}"
                )

                print(
                    f"Final pipeline P/R/F1: "
                    f"{final_metrics['precision']:.3f} / "
                    f"{final_metrics['recall']:.3f} / "
                    f"{final_metrics['f1']:.3f}"
                )

                timing_writer.writerow(
                    [
                        str(input_path),
                        model_name,
                        max_seq_len,
                        n,
                        cfg.top_k,

                        f"{total_tokenization_time:.6f}",
                        f"{total_embed_only_time:.6f}",
                        f"{total_similarity_time:.6f}",
                        f"{total_cross_encoder_time:.6f}",
                        f"{e2e_excl_tokenization_time:.6f}",
                        f"{wall_time:.6f}",

                        f"{tokenization_thr:.2f}",
                        f"{embedding_thr:.2f}",
                        f"{similarity_thr:.2f}",
                        f"{cross_query_thr:.2f}",
                        f"{cross_pair_thr:.2f}",
                        f"{e2e_excl_tok_thr:.2f}",
                        f"{wall_thr:.2f}",

                        total_rerouted_queries,
                        f"{rerouted_fraction:.6f}",
                        total_cross_encoder_pairs,

                        f"{cross_metrics['precision']:.6f}",
                        f"{cross_metrics['recall']:.6f}",
                        f"{cross_metrics['f1']:.6f}",

                        f"{final_metrics['precision']:.6f}",
                        f"{final_metrics['recall']:.6f}",
                        f"{final_metrics['f1']:.6f}",

                        int(cfg.run_cross_encoder and cross_encoder is not None),
                        cfg.cross_encoder_model or "",
                        cfg.cross_encoder_threshold,

                        f"{peak_allocated_mb:.2f}",
                        f"{peak_reserved_mb:.2f}",
                    ]
                )

                tfw.flush()
                dfw.flush()
                fw.flush()

                print(f"Wrote predictions: {out_path}")
                print(f"Wrote timings: {timing_path}")
                print(f"Wrote score details: {details_path}")

                eval_file(out_path)

    assert last_out_path is not None
    return last_out_path


def run_model_across_files(
    cfg: Config,
    *,
    model_path: str,
    input_files: Sequence[Path],
    label_desc: Dict[str, str],
    candidate_labels: Optional[Sequence[str]],
    train_labels: Optional[Set[str]],
) -> List[Path]:
    outputs: List[Path] = []
    base_model, _ = model_base_and_suffix(model_path)
    device = resolve_device(cfg.device)

    if "onnx" in base_model:
        encoder = SentenceTransformer(
            base_model,
            device=device,
            backend="onnx",
            model_kwargs={"provider": "CUDAExecutionProvider"},
        )
    else:
        encoder = SentenceTransformer(base_model, device=device)

    encoder.eval()

    if device.startswith("cuda") and "onnx" not in base_model:
        try:
            encoder[0].auto_model = torch.compile(
                encoder[0].auto_model,
                mode="reduce-overhead"
            )
            print("[INFO] torch.compile enabled")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")

    cross_encoder = None

    if cfg.run_cross_encoder:
        if not cfg.cross_encoder_model:
            raise ValueError("--run-cross-encoder requires --cross-encoder-model")

        print(f"[INFO] Loading cross-encoder: {cfg.cross_encoder_model}")

        cross_encoder = CrossEncoder(
            cfg.cross_encoder_model,
            device=device,
            max_length=cfg.cross_encoder_max_length,
        )

        if cfg.use_fp16 and device.startswith("cuda"):
            try:
                cross_encoder.model.half()
                print("[INFO] Cross-encoder FP16 enabled")
            except Exception as e:
                print(f"[WARN] Could not convert cross-encoder to FP16: {e}")

    all_test_labels = collect_all_test_labels(
        input_files,
        delimiter=cfg.input_delimiter,
        has_header=cfg.input_has_header,
        limit=cfg.limit_examples,
        sample_with_replacement=cfg.sample_with_replacement,
    )

    labels, emb_labels = build_retrieval_resources(
        cfg,
        model_path=model_path,
        encoder=encoder,
        label_desc=label_desc,
        candidate_labels=candidate_labels,
        train_labels=train_labels,
        all_test_labels=all_test_labels,
    )

    for input_path in input_files:
        outputs.append(
            run_retrieval_model_on_file(
                cfg,
                encoder=encoder,
                cross_encoder=cross_encoder,
                model_path=model_path,
                input_path=input_path,
                labels=labels,
                label_desc=label_desc,
                emb_labels=emb_labels,
            )
        )

    return outputs


def load_fasttext_module(fasttext_path: Path):
    fasttext_path = fasttext_path.resolve()

    if str(fasttext_path) not in sys.path:
        sys.path.insert(0, str(fasttext_path))

    try:
        import fasttext
    except ImportError as exc:
        raise ImportError(
            f"Could not import fasttext from {fasttext_path}. "
            f"Make sure fastText Python bindings are installed or built."
        ) from exc

    return fasttext


def clean_for_fasttext(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ").strip()


def normalize_np(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return x / denom


# def encode_fasttext_texts(
#     model,
#     texts: Sequence[str],
#     *,
#     batch_size: int,
#     normalize: bool = True,
# ) -> Tuple[np.ndarray, float]:
#     dim = model.get_dimension()
#     vectors: List[np.ndarray] = []
#
#     start = time.perf_counter()
#
#     for i in range(0, len(texts), batch_size):
#         batch = texts[i:i + batch_size]
#         arr = np.empty((len(batch), dim), dtype=np.float32)
#
#         for j, text in enumerate(batch):
#             arr[j] = model.get_sentence_vector(clean_for_fasttext(text))
#
#         if normalize:
#             arr = normalize_np(arr).astype(np.float32)
#
#         vectors.append(arr)
#
#     elapsed = time.perf_counter() - start
#     return np.vstack(vectors), elapsed

def classify_fasttext_texts(
    model,
    texts,
    *,
    batch_size: int,
    k: int = 1,
):
    all_labels = []
    all_probs = []

    start = time.perf_counter()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        for text in batch:
            labels, probs = model.predict(
                clean_for_fasttext(text),
                k=k,
            )

            all_labels.append(list(labels))
            all_probs.append(np.asarray(probs, dtype=np.float32).tolist())

    elapsed = time.perf_counter() - start
    return all_labels, all_probs, elapsed

def run_fasttext_on_file(
    cfg: Config,
    *,
    input_path: Path,
    label_desc: Dict[str, str],
    candidate_labels: Optional[Sequence[str]],
    train_labels: Optional[Set[str]],
) -> Path:
    print(f"[FASTTEXT] Loading module from {cfg.fasttext_path}")
    fasttext = load_fasttext_module(cfg.fasttext_path)

    print(f"[FASTTEXT] Loading model: {cfg.fasttext_model}")
    ft_model = fasttext.load_model(str(cfg.fasttext_model))

    device = resolve_device(cfg.device)

    examples = load_examples(
        input_path,
        delimiter=cfg.input_delimiter,
        has_header=cfg.input_has_header,
        limit=cfg.limit_examples,
        sample_with_replacement=cfg.sample_with_replacement,
    )

    golds = ['__label__'+ex.gold.split('.')[0] for ex in examples]
    texts = [ex.text for ex in examples]

    #labels = build_candidate_labels(
    #    mode=cfg.candidate_mode,
    #    label_desc=label_desc,
    #    test_labels=golds,
    #    train_labels=train_labels,
    #    candidate_labels=candidate_labels,
    #)

    #descriptions = [label_desc[label] for label in labels]

    #print(f"[FASTTEXT] Candidate labels: {len(labels):,}")
    print(f"[FASTTEXT] Examples: {len(texts):,}")

    #desc_embs, desc_time = encode_fasttext_texts(
    #    ft_model,
    #    descriptions,
    #    batch_size=cfg.fasttext_batch_size,
    #    normalize=True,
    #)

    #print(
    #    f"[FASTTEXT] Embedded {len(descriptions):,} descriptions "
    #    f"in {desc_time:.2f}s "
    #    f"({len(descriptions) / desc_time:.1f} desc/s)"
    #)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / f"{cfg.output_prefix}_{input_path.stem}_fasttext.tsv"

    #desc_torch = torch.from_numpy(desc_embs)

    #if device.startswith("cuda"):
    #    desc_torch = desc_torch.to(device, non_blocking=True)

    total_embed_time = 0.0
    total_similarity_time = 0.0
    wall_start = time.perf_counter()

    with out_path.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw, delimiter="\t")
        for i in tqdm.tqdm(
            range(0, len(texts), cfg.fasttext_batch_size),
            desc=f"fastText::{input_path.name}",
        ):
            batch_texts = texts[i:i + cfg.fasttext_batch_size]
            batch_golds = golds[i:i + cfg.fasttext_batch_size]
            pred_labels, pred_probs, classify_time = classify_fasttext_texts(
                ft_model,
                batch_texts,
                batch_size=cfg.fasttext_batch_size,
                k=50,
            )
            total_embed_time += classify_time
            for gold, ids, vals in zip(batch_golds, pred_labels, pred_probs):
                row = [gold] + [
                    f"{label}~{score}"
                    for label, score in zip(ids, vals)
                ]
                writer.writerow(row)
            if classify_time > 0:
                print(
                    f"[FASTTEXT CHUNK] {len(batch_texts):,} texts | "
                    f"classify={classify_time:.2f}s | "
                    f"classify_throughput={len(batch_texts) / classify_time:.1f} texts/s"
                )
    wall_time = time.perf_counter() - wall_start
    n = len(texts)
    print()
    print(f"[FASTTEXT] Finished {input_path}")
    print(f"Examples: {n:,}")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"Embedding time: {total_embed_time:.2f}s")
    print(f"Similarity time: {total_similarity_time:.2f}s")
    if total_embed_time > 0:
        print(f"Embedding throughput: {n / total_embed_time:.1f} texts/s")
    if total_similarity_time > 0:
        print(f"Similarity throughput: {n / total_similarity_time:.1f} texts/s")
    if wall_time > 0:
        print(f"End-to-end throughput: {n / wall_time:.1f} texts/s")
    print(f"Wrote: {out_path}")
    eval_file(out_path)
    return out_path


def run(cfg: Config) -> List[Path]:
    print("Run started")
    outputs: List[Path] = []
    train_labels = load_train_labels(cfg)
    candidate_labels = load_candidate_labels(cfg)

    desc_cache: Dict[str, Dict[str, str]] = {}

    for model_path in cfg.models:
        print(f"[INFO] Loading LaBSE/SentenceTransformer model: {model_path}")
        description = get_description_type(model_path)
        if description not in desc_cache:
            desc_cache[description] = load_label_descriptions(description)
        outputs.extend(
            run_model_across_files(
                cfg,
                model_path=model_path,
                input_files=cfg.input_files,
                label_desc=desc_cache[description],
                candidate_labels=candidate_labels,
                train_labels=train_labels,
            )
        )
    if cfg.run_fasttext:
        print("[INFO] Running fastText benchmark")
        #description = "label"
        #if description not in desc_cache:
        #    desc_cache[description] = load_label_descriptions(description)
        for input_path in cfg.input_files:
            outputs.append(
                run_fasttext_on_file(
                    cfg,
                    input_path=input_path,
                    label_desc=desc_cache[description],
                    candidate_labels=candidate_labels,
                    train_labels=train_labels,
                )
            )
    return outputs


def parse_args() -> Config:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--models",
        nargs="+",
        default=[
            "./models/GlotLID-10M_desc",
            #"./models/GlotLID-10M_desc-onnx",
            #"./models/GlotLID-10M_desc-onnx-int8"
        ],
    )

    p.add_argument(
        "--input-files",
        nargs="+",
        type=Path,
        default=[Path("data/flores200/Flores_DEV.tsv")],
    )

    p.add_argument("--train_ds", default="lid201")
    p.add_argument("--out-dir", type=Path, default=Path("./results"))

    p.add_argument("--datasets_stats_pkl", type=Path, default=Path("./resources/all_stats.pkl"))

    p.add_argument("--input-delimiter", default="\t")
    p.add_argument("--input-has-header", action="store_true")

    p.add_argument("--candidate-labels-tsv", type=Path, default=None)
    p.add_argument("--candidate-label-col", type=int, default=0)
    p.add_argument("--candidate-delimiter", default="\t")
    p.add_argument("--candidate-has-header", action="store_true")

    p.add_argument(
        "--candidate-mode",
        choices=[x.value for x in CandidateMode],
        default=CandidateMode.TRAIN_ONLY.value,
    )

    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--query-chunk-size", type=int, default=100_000)

    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=30_000)

    p.add_argument("--desc-max-length", type=int, default=64)
    p.add_argument("--query-max-length", type=int, default=64)

    p.add_argument("--desc-batch-size", type=int, default=None)
    p.add_argument("--query-batch-size", type=int, default=None)

    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--output-prefix", default="preds")

    p.add_argument("--use-fp16", action="store_true")
    p.add_argument("--no-use-fp16", dest="use_fp16", action="store_false")
    p.set_defaults(use_fp16=True)

    p.add_argument("--limit-examples", type=int)
    p.add_argument("--no-sample-with-replacement", dest="sample_with_replacement", action="store_true")
    p.set_defaults(sample_with_replacement=True)

    p.add_argument("--run-fasttext", action="store_true")
    p.add_argument("--no-run-fasttext", dest="run_fasttext", action="store_false")
    p.set_defaults(run_fasttext=True)

    p.add_argument("--fasttext-path", type=Path, default=Path("../fastText"))
    p.add_argument("--fasttext-model", type=Path, default=Path("../fastText/model_lid201-corpus_sampled10M.bin"))
    p.add_argument("--fasttext-batch-size", type=int, default=100_000)

    p.add_argument("--run-cross-encoder", default="store_true")
    p.add_argument("--no-run-cross-encoder", dest="run_cross_encoder", action="store_false")
    p.set_defaults(run_cross_encoder=True)

    p.add_argument(
        "--cross-encoder-model",
        default="./models/cross_encoder_udhr-lid",
        help="Cross-encoder used to rerank low-confidence bi-encoder predictions.",
    )

    p.add_argument(
        "--cross-encoder-threshold",
        type=float,
        default=0.5,
        help="Activate cross-encoder when the bi-encoder top-1 score is below this threshold.",
    )

    p.add_argument(
        "--cross-encoder-batch-size",
        type=int,
        default=256,
        help="Batch size for cross-encoder pair scoring.",
    )

    p.add_argument(
        "--cross-encoder-max-length",
        type=int,
        default=512,
        help="Maximum sequence length for cross-encoder query-description pairs.",
    )

    args = p.parse_args()

    return Config(
        models=args.models,
        input_files=args.input_files,
        train_ds=args.train_ds,
        out_dir=args.out_dir,
        input_delimiter=args.input_delimiter,
        input_has_header=args.input_has_header,
        candidate_labels_tsv=args.candidate_labels_tsv,
        datasets_stats_pkl=args.datasets_stats_pkl,
        candidate_label_col=args.candidate_label_col,
        candidate_delimiter=args.candidate_delimiter,
        candidate_has_header=args.candidate_has_header,
        candidate_mode=CandidateMode(args.candidate_mode),
        top_k=args.top_k,
        query_chunk_size=args.query_chunk_size,
        device=args.device,
        batch_size=args.batch_size,
        desc_max_length=args.desc_max_length,
        query_max_length=args.query_max_length,
        desc_batch_size=args.desc_batch_size,
        query_batch_size=args.query_batch_size,
        cache_dir=args.cache_dir,
        output_prefix=args.output_prefix,
        use_fp16=args.use_fp16,
        limit_examples=args.limit_examples,
        sample_with_replacement=args.sample_with_replacement,
        run_fasttext=args.run_fasttext,
        fasttext_path=args.fasttext_path,
        fasttext_model=args.fasttext_model,
        fasttext_batch_size=args.fasttext_batch_size,
        run_cross_encoder=args.run_cross_encoder,
        cross_encoder_model=args.cross_encoder_model,
        cross_encoder_threshold=args.cross_encoder_threshold,
        cross_encoder_batch_size=args.cross_encoder_batch_size,
        cross_encoder_max_length=args.cross_encoder_max_length,
    )


def main() -> None:
    cfg = parse_args()
    device = resolve_device(cfg.device)

    print(f"Device resolved to: {device} cuda_available={torch.cuda.is_available()}")
    print(f"FP16 enabled: {cfg.use_fp16 and device.startswith('cuda')}")
    print(f"LaBSE query batch size: {cfg.query_batch_size or cfg.batch_size}")
    print(f"LaBSE query max length: {cfg.query_max_length}")
    print(f"fastText enabled: {cfg.run_fasttext}")
    print(f"fastText model: {cfg.fasttext_model}")

    print(f"Cross-encoder enabled: {cfg.run_cross_encoder}")
    print(f"Cross-encoder model: {cfg.cross_encoder_model}")
    print(f"Cross-encoder threshold: {cfg.cross_encoder_threshold}")
    print(f"Cross-encoder batch size: {cfg.cross_encoder_batch_size}")
    print(f"Cross-encoder max length: {cfg.cross_encoder_max_length}")

    configure_torch_for_inference(device)
    run(cfg)


if __name__ == "__main__":
    main()