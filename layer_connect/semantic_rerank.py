from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer Connect cross-encoder worker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or not all(
        isinstance(item, list)
        and len(item) == 2
        and all(isinstance(value, str) for value in item)
        for item in pairs
    ):
        raise ValueError("reranker input은 [query, passage] 목록이어야 합니다.")
    model = CrossEncoder(
        args.model,
        device=args.device,
        max_length=max(32, args.max_length),
        local_files_only=args.offline,
    )
    scores = model.predict(
        pairs,
        batch_size=max(1, args.batch_size),
        show_progress_bar=False,
        activation_fn=torch.nn.Sigmoid(),
        convert_to_numpy=True,
    )
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.shape != (len(pairs),) or not np.isfinite(scores).all():
        raise ValueError(f"잘못된 reranker score shape/value: {scores.shape}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, scores, allow_pickle=False)
    print(
        json.dumps(
            {
                "model": args.model,
                "pairs": len(pairs),
                "min": float(scores.min(initial=0.0)),
                "max": float(scores.max(initial=0.0)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
