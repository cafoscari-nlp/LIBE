from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_SENTENCE = "Пур халӑх та тивӗҫлӗ тата хисепре пурӑнма пӗр тан праваллӑ."
MAX_LENGTH = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/GlotLID-10M_desc"))
    parser.add_argument("--desc-tsv", type=Path, default=Path("resources/WorldsLangs.tsv"))
    parser.add_argument("--sentence", nargs="+", default=[DEFAULT_SENTENCE])
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for inference (auto: cuda > mps > cpu).",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_label_descriptions(path: Path) -> Tuple[List[str], List[str]]:
    """Read rows `label<TAB>description` from the WorldsLangs TSV."""
    labels: List[str] = []
    descriptions: List[str] = []

    with path.open("r", encoding="utf-8") as fr:
        for line in fr:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                labels.append(parts[0])
                descriptions.append(parts[1])

    return labels, descriptions


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    def embed(texts: List[str]) -> torch.Tensor:
        chunks: List[torch.Tensor] = []

        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
            ).to(device)
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                chunks.append(torch.nn.functional.normalize(pooled, p=2, dim=1))

        return torch.cat(chunks, dim=0)

    labels, descriptions = load_label_descriptions(args.desc_tsv)

    scores = embed(args.sentence) @ embed(descriptions).t()
    top = scores.topk(min(args.top_k, len(labels)), dim=-1)

    for row, sentence in enumerate(args.sentence):
        print(sentence)
        for score, idx in zip(top.values[row].tolist(), top.indices[row].tolist()):
            print(f"{score:.4f}\t{labels[idx]}")


if __name__ == "__main__":
    main()
