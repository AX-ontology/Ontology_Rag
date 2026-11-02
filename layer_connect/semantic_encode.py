from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer Connect semantic embedding worker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    texts = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(texts, list) or not all(isinstance(item, str) for item in texts):
        raise ValueError("embedding input은 문자열 목록이어야 합니다.")
    if not texts:
        raise ValueError("embedding input이 비어 있습니다.")
    model = SentenceTransformer(
        args.model,
        device=args.device,
        local_files_only=args.offline,
    )
    vectors = model.encode(
        texts,
        batch_size=max(1, args.batch_size),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError(f"잘못된 embedding shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("embedding에 NaN/Inf가 포함되어 있습니다.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, vectors, allow_pickle=False)
    print(
        json.dumps(
            {
                "model": args.model,
                "device": args.device,
                "texts": len(texts),
                "dimension": int(vectors.shape[1]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
