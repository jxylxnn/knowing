from pathlib import Path
import argparse

from src.contracts.artifacts import ArtifactContract, validate_runtime_artifacts
from src.contracts.projections import validate_projection_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--projection-csv")
    parser.add_argument("--transformer-required", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    validate_runtime_artifacts(
        ArtifactContract(
            models_dir=Path(args.models_dir),
            transformer_required=args.transformer_required,
        )
    )

    if args.projection_csv:
        validate_projection_csv(Path(args.projection_csv))

    print("✅ Contracts passed")


if __name__ == "__main__":
    main()
