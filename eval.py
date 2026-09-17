#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix


# -------------------------
# Model pretraining language groups
# -------------------------

LABSE_ISO3: Set[str] = {
    "afr", "amh", "ara", "asm", "aze", "bel", "bul", "ben", "bod", "bos",
    "cat", "ceb", "cos", "ces", "cym", "dan", "deu", "ell", "eng", "epo",
    "spa", "est", "eus", "fas", "fin", "fra", "fry", "gle", "gla", "glg",
    "guj", "hau", "haw", "heb", "hin", "hmn", "hrv", "hat", "hun", "hye",
    "ind", "ibo", "isl", "ita", "jpn", "jav", "kat", "kaz", "khm", "kan",
    "kor", "kur", "kir", "lat", "ltz", "lao", "lit", "lav", "mlg", "mri",
    "mkd", "mal", "mon", "mar", "msa", "mlt", "mya", "nep", "nld", "nor",
    "nya", "ori", "pan", "pol", "por", "ron", "rus", "kin", "sin", "slk",
    "slv", "smo", "sna", "som", "sqi", "srp", "sot", "sun", "swe", "swa",
    "tam", "tel", "tgk", "tha", "tuk", "tgl", "tur", "tat", "uig", "ukr",
    "urd", "uzb", "vie", "wol", "xho", "yid", "yor", "zho", "zul",
}

MINILM_ISO3: Set[str] = {
    "ara", "bul", "cat", "ces", "dan", "deu", "ell", "eng", "spa", "est",
    "fas", "fin", "fra", "glg", "guj", "heb", "hin", "hrv", "hun", "hye",
    "ind", "ita", "jpn", "kat", "kor", "kur", "lit", "lav", "mkd", "mon",
    "mar", "msa", "mya", "nor", "nld", "pol", "por", "ron", "rus", "slk",
    "slv", "sqi", "srp", "swe", "tha", "tur", "ukr", "urd", "vie",
}

E5_XLMR_ISO3: Set[str] = {
    "afr", "amh", "ara", "asm", "aze", "bel", "bul", "ben", "bos", "cat",
    "ceb", "ces", "cym", "dan", "deu", "ell", "eng", "spa", "est", "eus",
    "fas", "fin", "fra", "gle", "glg", "guj", "hau", "heb", "hin", "hrv",
    "hun", "hye", "ind", "ibo", "isl", "ita", "jpn", "jav", "kat", "kaz",
    "khm", "kan", "kor", "kur", "kir", "ltz", "lao", "lit", "lav", "mlg",
    "mkd", "mal", "mon", "mar", "msa", "mya", "nep", "nld", "nor", "nya",
    "ori", "pan", "pol", "por", "ron", "rus", "sin", "slk", "slv", "som",
    "sqi", "srp", "swa", "swe", "tam", "tel", "tgk", "tha", "tgl", "tur",
    "ukr", "urd", "uzb", "vie", "wol", "xho", "yid", "yor", "zho", "zul",
}

DISTILUSE_ISO3: Set[str] = {
    "ara", "zho", "nld", "eng", "fra", "deu", "ita", "kor", "pol", "por",
    "rus", "spa", "tur",
}

MONO_ISO3: Set[str] = {"eng"}
RAND_ISO3: Set[str] = set()


# -------------------------
# Data structures
# -------------------------

@dataclass(frozen=True)
class EvalConfig:
    pred_files: List[Path]
    pred_glob: Optional[str]
    results_dir: Path
    thresholds: Sequence[float]
    reject_label: str
    threshold_inclusive: bool
    realistic: bool
    json_out: Optional[Path]
    summary_tsv: Optional[Path]
    per_language_tsv_dir: Optional[Path]
    seen_group: str


@dataclass
class PerClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    fpr: float
    support: int


@dataclass
class GroupMetrics:
    name: str
    num_labels: int
    macro_precision: float
    macro_recall: float
    macro_f1: float
    macro_fpr: float
    support: int


@dataclass
class FileResult:
    pred_file: str
    threshold: float
    realistic: bool
    labels: List[str]
    macro: dict
    micro_accuracy: float
    coverage: float
    accepted: int
    total: int
    acc_on_accepted: float
    seen_macro: dict
    unseen_macro: dict
    per_class: List[dict]


# -------------------------
# Basic utilities
# -------------------------

def normalize_label(label: str) -> str:
    label = str(label).strip()

    if label == "__REJECT__":
        return label

    if "REJECT" in label:
        return "__REJECT__"

    label = label.replace("__label__", "")
    return label.split("_")[0]


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(
        num,
        den,
        out=np.zeros_like(num, dtype=float),
        where=den != 0,
    )


def parse_pred_cell(cell: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Parse a cell of the form:

      label~score

    Returns:
      (label, score)

    If the cell has no score, returns:
      (label, None)
    """
    cell = cell.strip()

    if not cell:
        return None, None

    if "~" not in cell:
        return normalize_label(cell), None

    try:
        label, score_s = cell.rsplit("~", 1)
        return normalize_label(label), float(score_s)
    except Exception:
        return None, None


def read_pred_tsv(
    path: Path,
) -> Tuple[List[str], List[List[Tuple[Optional[str], Optional[float]]]]]:
    golds: List[str] = []
    preds: List[List[Tuple[Optional[str], Optional[float]]]] = []

    with path.open("r", encoding="utf-8") as fr:
        reader = csv.reader(fr, delimiter="\t")

        for row in reader:
            if not row:
                continue

            golds.append(normalize_label(row[0]))
            preds.append([parse_pred_cell(cell) for cell in row[1:]])

    return golds, preds


def discover_prediction_files(
    *,
    explicit_files: Sequence[Path],
    pred_glob: Optional[str],
    results_dir: Path,
) -> List[Path]:
    files: List[Path] = []

    files.extend(explicit_files)

    if pred_glob is not None:
        files.extend(sorted(results_dir.glob(pred_glob)))

    deduped = []
    seen = set()

    for path in files:
        path = Path(path)

        if path not in seen:
            deduped.append(path)
            seen.add(path)

    return deduped


# -------------------------
# Seen/unseen grouping
# -------------------------

def infer_seen_languages(pred_path: str, group: str) -> Set[str]:
    group = group.lower()
    path_lower = pred_path.lower()

    if group == "none":
        return set()

    if group == "labse":
        return LABSE_ISO3

    if group == "minilm":
        return MINILM_ISO3

    if group == "e5":
        return E5_XLMR_ISO3

    if group == "distiluse":
        return DISTILUSE_ISO3

    if group == "mono":
        return MONO_ISO3

    if group == "rand":
        return RAND_ISO3

    if group != "auto":
        raise ValueError(f"Unknown seen-language group: {group}")

    if "minilm" in path_lower:
        return MINILM_ISO3

    if "e5" in path_lower:
        return E5_XLMR_ISO3

    if "distiluse" in path_lower:
        return DISTILUSE_ISO3

    if "rand" in path_lower:
        return RAND_ISO3

    if "bert" in path_lower or "fasttext" in path_lower:
        return MONO_ISO3

    if (
        "labse" in path_lower
        or "gemma" in path_lower
        or "glotlid" in path_lower
        or "lid201" in path_lower
        or "libe" in path_lower
    ):
        return LABSE_ISO3

    print(
        f"[WARN] Could not infer seen-language group for {pred_path}. "
        f"Using empty seen set.",
        flush=True,
    )
    return set()


# -------------------------
# Prediction choice
# -------------------------

def choose_final_pred(
    candidates: Sequence[Tuple[Optional[str], Optional[float]]],
    *,
    threshold: float,
    reject_label: str,
    inclusive: bool,
    realistic: bool,
    gold_set: Optional[Set[str]],
) -> str:
    """
    realistic=True:
      Use top-1 prediction as-is.

    realistic=False:
      Filter predictions to labels appearing in the gold label set, then use
      the first remaining candidate. This is useful for controlled closed-set
      analysis but is not a realistic deployment setting.
    """
    if not candidates:
        return reject_label

    if realistic:
        label, score = candidates[0]
    else:
        if gold_set is None:
            return reject_label

        filtered = [
            (label, score)
            for label, score in candidates
            if label is not None and normalize_label(label) in gold_set
        ]

        if not filtered:
            return reject_label

        label, score = filtered[0]

    if label is None or score is None:
        return reject_label

    passes = score >= threshold if inclusive else score > threshold

    return normalize_label(label) if passes else reject_label


# -------------------------
# Metrics
# -------------------------

def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    reject_label: str,
) -> Tuple[List[str], dict, List[PerClassMetrics], float]:
    true_labels = sorted(set(y_true))
    labels = true_labels.copy()

    if reject_label not in labels:
        labels.append(reject_label)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - (tp + fp + fn)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)
    support = cm.sum(axis=1)

    real_mask = np.array([label != reject_label for label in labels], dtype=bool)

    if real_mask.any():
        macro = {
            "precision": float(precision[real_mask].mean()),
            "recall": float(recall[real_mask].mean()),
            "f1": float(f1[real_mask].mean()),
            "fpr": float(fpr[real_mask].mean()),
        }
    else:
        macro = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "fpr": 0.0,
        }

    per_class = [
        PerClassMetrics(
            label=label,
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            fpr=float(fpr[i]),
            support=int(support[i]),
        )
        for i, label in enumerate(labels)
    ]

    micro_accuracy = float(
        sum(1 for gold, pred in zip(y_true, y_pred) if gold == pred) / len(y_true)
    ) if y_true else 0.0

    return labels, macro, per_class, micro_accuracy


def aggregate_group_metrics(
    per_class: Sequence[PerClassMetrics],
    *,
    labels_in_group: Set[str],
    reject_label: str,
    name: str,
) -> GroupMetrics:
    rows = [
        row
        for row in per_class
        if row.label != reject_label and row.label in labels_in_group
    ]

    if not rows:
        return GroupMetrics(
            name=name,
            num_labels=0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            macro_fpr=0.0,
            support=0,
        )

    return GroupMetrics(
        name=name,
        num_labels=len(rows),
        macro_precision=float(np.mean([row.precision for row in rows])),
        macro_recall=float(np.mean([row.recall for row in rows])),
        macro_f1=float(np.mean([row.f1 for row in rows])),
        macro_fpr=float(np.mean([row.fpr for row in rows])),
        support=int(sum(row.support for row in rows)),
    )


# -------------------------
# Output writers
# -------------------------

def write_per_language_tsv(
    out_path: Path,
    *,
    pred_file: str,
    threshold: float,
    realistic: bool,
    per_class: Sequence[PerClassMetrics],
    reject_label: str,
    seen_languages: Set[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw, delimiter="\t")

        writer.writerow([
            "pred_file",
            "threshold",
            "realistic",
            "label",
            "seen_in_model_pretraining",
            "support",
            "precision",
            "recall",
            "f1",
            "fpr",
        ])

        for row in per_class:
            if row.label == reject_label:
                continue

            writer.writerow([
                pred_file,
                threshold,
                realistic,
                row.label,
                row.label in seen_languages,
                row.support,
                f"{row.precision:.6f}",
                f"{row.recall:.6f}",
                f"{row.f1:.6f}",
                f"{row.fpr:.6f}",
            ])


def write_summary_tsv(
    out_path: Path,
    results: Sequence[FileResult],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.writer(fw, delimiter="\t")

        writer.writerow([
            "pred_file",
            "threshold",
            "realistic",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "macro_fpr",
            "micro_accuracy",
            "coverage",
            "accepted",
            "total",
            "acc_on_accepted",
            "seen_num_labels",
            "seen_macro_f1",
            "unseen_num_labels",
            "unseen_macro_f1",
        ])

        for row in results:
            writer.writerow([
                row.pred_file,
                row.threshold,
                row.realistic,
                f"{row.macro['precision']:.6f}",
                f"{row.macro['recall']:.6f}",
                f"{row.macro['f1']:.6f}",
                f"{row.macro['fpr']:.6f}",
                f"{row.micro_accuracy:.6f}",
                f"{row.coverage:.6f}",
                row.accepted,
                row.total,
                f"{row.acc_on_accepted:.6f}",
                row.seen_macro["num_labels"],
                f"{row.seen_macro['f1']:.6f}",
                row.unseen_macro["num_labels"],
                f"{row.unseen_macro['f1']:.6f}",
            ])


# -------------------------
# Main evaluation
# -------------------------

def evaluate_file_at_threshold(
    *,
    pred_path: Path,
    threshold: float,
    cfg: EvalConfig,
) -> FileResult:
    golds, all_candidates = read_pred_tsv(pred_path)
    gold_set = set(golds)

    y_true: List[str] = []
    y_pred: List[str] = []

    for gold, candidates in zip(golds, all_candidates):
        final = choose_final_pred(
            candidates,
            threshold=threshold,
            reject_label=cfg.reject_label,
            inclusive=cfg.threshold_inclusive,
            realistic=cfg.realistic,
            gold_set=gold_set,
        )

        y_true.append(normalize_label(gold))
        y_pred.append(normalize_label(final))

    labels, macro, per_class, micro_accuracy = compute_metrics(
        y_true,
        y_pred,
        cfg.reject_label,
    )

    seen_languages = infer_seen_languages(str(pred_path), cfg.seen_group)

    seen_labels = {
        row.label
        for row in per_class
        if row.label != cfg.reject_label and row.label in seen_languages
    }

    unseen_labels = {
        row.label
        for row in per_class
        if row.label != cfg.reject_label and row.label not in seen_languages
    }

    seen_group = aggregate_group_metrics(
        per_class,
        labels_in_group=seen_labels,
        reject_label=cfg.reject_label,
        name="seen",
    )

    unseen_group = aggregate_group_metrics(
        per_class,
        labels_in_group=unseen_labels,
        reject_label=cfg.reject_label,
        name="unseen",
    )

    accepted_mask = [pred != cfg.reject_label for pred in y_pred]
    num_accepted = sum(accepted_mask)
    total = len(y_pred)
    coverage = num_accepted / total if total else 0.0

    num_correct_accepted = sum(
        1
        for gold, pred in zip(y_true, y_pred)
        if pred != cfg.reject_label and pred == gold
    )

    acc_on_accepted = (
        num_correct_accepted / num_accepted
        if num_accepted
        else 0.0
    )

    result = FileResult(
        pred_file=str(pred_path),
        threshold=float(threshold),
        realistic=cfg.realistic,
        labels=labels,
        macro=macro,
        micro_accuracy=micro_accuracy,
        coverage=float(coverage),
        accepted=int(num_accepted),
        total=int(total),
        acc_on_accepted=float(acc_on_accepted),
        seen_macro={
            "precision": seen_group.macro_precision,
            "recall": seen_group.macro_recall,
            "f1": seen_group.macro_f1,
            "fpr": seen_group.macro_fpr,
            "num_labels": seen_group.num_labels,
            "support": seen_group.support,
        },
        unseen_macro={
            "precision": unseen_group.macro_precision,
            "recall": unseen_group.macro_recall,
            "f1": unseen_group.macro_f1,
            "fpr": unseen_group.macro_fpr,
            "num_labels": unseen_group.num_labels,
            "support": unseen_group.support,
        },
        per_class=[asdict(row) for row in per_class],
    )

    if cfg.per_language_tsv_dir is not None:
        out_name = (
            f"{pred_path.stem}"
            f"__thr_{str(threshold).replace('.', 'p')}"
            f"__per_language.tsv"
        )

        write_per_language_tsv(
            cfg.per_language_tsv_dir / out_name,
            pred_file=str(pred_path),
            threshold=threshold,
            realistic=cfg.realistic,
            per_class=per_class,
            reject_label=cfg.reject_label,
            seen_languages=seen_languages,
        )

    return result


def print_summary(result: FileResult) -> None:
    print(
        f"{Path(result.pred_file).stem}\t"
        f"realistic={result.realistic}\t"
        f"T={result.threshold}\n"
        f"P={result.macro['precision']:.3f}\t"
        f"R={result.macro['recall']:.3f}\t"
        f"F1={result.macro['f1']:.3f}\t"
        f"FPR={result.macro['fpr']:.4f}\t"
        f"micro_acc={result.micro_accuracy:.3f}\n"
        f"coverage={result.coverage:.3f}\t"
        f"accepted={result.accepted}/{result.total}\t"
        f"acc_on_accepted={result.acc_on_accepted:.3f}\n"
        f"SEEN\tn={result.seen_macro['num_labels']}\t"
        f"support={result.seen_macro['support']}\t"
        f"P={result.seen_macro['precision']:.3f}\t"
        f"R={result.seen_macro['recall']:.3f}\t"
        f"F1={result.seen_macro['f1']:.3f}\t"
        f"FPR={result.seen_macro['fpr']:.4f}\n"
        f"UNSEEN\tn={result.unseen_macro['num_labels']}\t"
        f"support={result.unseen_macro['support']}\t"
        f"P={result.unseen_macro['precision']:.3f}\t"
        f"R={result.unseen_macro['recall']:.3f}\t"
        f"F1={result.unseen_macro['f1']:.3f}\t"
        f"FPR={result.unseen_macro['fpr']:.4f}\n",
        flush=True,
    )


def evaluate_files(cfg: EvalConfig) -> List[FileResult]:
    pred_files = discover_prediction_files(
        explicit_files=cfg.pred_files,
        pred_glob=cfg.pred_glob,
        results_dir=cfg.results_dir,
    )

    if not pred_files:
        raise ValueError(
            "No prediction files found. Provide --pred-files or --pred-glob."
        )

    results: List[FileResult] = []

    for pred_path in pred_files:
        if not pred_path.exists():
            print(f"[WARN] Prediction file not found: {pred_path}; skipping.")
            continue

        for threshold in cfg.thresholds:
            result = evaluate_file_at_threshold(
                pred_path=pred_path,
                threshold=threshold,
                cfg=cfg,
            )

            results.append(result)
            print_summary(result)

    return results


# -------------------------
# CLI
# -------------------------

def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate prediction TSVs produced by run_lid_evaluation.py. "
            "Each row must be: gold<TAB>pred1~score<TAB>pred2~score..."
        )
    )

    parser.add_argument(
        "--pred-files",
        nargs="*",
        type=Path,
        default=[],
        help="Prediction TSV files to evaluate.",
    )

    parser.add_argument(
        "--pred-glob",
        default=None,
        help=(
            "Optional glob pattern under --results-dir, "
            "for example 'preds_Flores_DEV_*.tsv'."
        ),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        help="Directory used with --pred-glob.",
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.0],
        help="Score thresholds to evaluate.",
    )

    parser.add_argument(
        "--reject-label",
        default="__REJECT__",
        help="Label used when no candidate passes the threshold.",
    )

    parser.add_argument(
        "--inclusive",
        action="store_true",
        help="Use score >= threshold. Default uses score > threshold.",
    )

    parser.add_argument(
        "--realistic",
        action="store_true",
        help=(
            "Use top-1 prediction as-is. "
            "Without this flag, candidates are first filtered to the gold-label set."
        ),
    )

    parser.add_argument(
        "--seen-group",
        default="auto",
        choices=[
            "auto",
            "labse",
            "e5",
            "minilm",
            "distiluse",
            "mono",
            "rand",
            "none",
        ],
        help=(
            "Language group used for seen/unseen aggregation. "
            "auto infers from the prediction filename."
        ),
    )

    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full JSON results.",
    )

    parser.add_argument(
        "--summary-tsv",
        type=Path,
        default=None,
        help="Optional path to write compact summary TSV results.",
    )

    parser.add_argument(
        "--per-language-tsv-dir",
        type=Path,
        default=None,
        help="Optional directory for per-language TSV metrics.",
    )

    args = parser.parse_args()

    return EvalConfig(
        pred_files=args.pred_files,
        pred_glob=args.pred_glob,
        results_dir=args.results_dir,
        thresholds=args.thresholds,
        reject_label=args.reject_label,
        threshold_inclusive=args.inclusive,
        realistic=args.realistic,
        json_out=args.json_out,
        summary_tsv=args.summary_tsv,
        per_language_tsv_dir=args.per_language_tsv_dir,
        seen_group=args.seen_group,
    )


def main() -> None:
    cfg = parse_args()
    results = evaluate_files(cfg)

    if cfg.json_out is not None:
        cfg.json_out.parent.mkdir(parents=True, exist_ok=True)

        with cfg.json_out.open("w", encoding="utf-8") as fo:
            json.dump(
                [asdict(result) for result in results],
                fo,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Wrote JSON results to {cfg.json_out}", flush=True)

    if cfg.summary_tsv is not None:
        write_summary_tsv(cfg.summary_tsv, results)
        print(f"Wrote summary TSV to {cfg.summary_tsv}", flush=True)


if __name__ == "__main__":
    main()