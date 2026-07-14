from __future__ import annotations

import argparse
from pathlib import Path

from app.tools.embedding_tools import DEFAULT_BGE_MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and save BGE locally.")
    parser.add_argument("--model-name", default=DEFAULT_BGE_MODEL_NAME)
    parser.add_argument(
        "--output-dir",
        default="data/models/bge-small-en-v1.5",
        help="Local directory used later as BGE_MODEL_PATH.",
    )
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.model_name)
    model.save(str(output_dir))
    print(f"Saved {args.model_name} to {output_dir}")


if __name__ == "__main__":
    main()
