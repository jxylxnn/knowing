from pathlib import Path
import argparse

from src.contracts.artifacts import ArtifactContract, validate_runtime_artifacts
from src.contracts.projections import validate_projection_csv
from src.contracts.runtime_status import resolve_active_models_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--projection-csv")
    parser.add_argument("--transformer-required", action="store_true")
    parser.add_argument(
        "--allow-legacy-artifacts",
        action="store_true",
        help="Allow quarantined pre-feature_schema_v4 artifacts; never use for promotion/replay.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.allow_legacy_artifacts:
        print("UNSAFE LEGACY ARTIFACT MODE: replay and promotion are disabled")

    supplied_models_dir = Path(args.models_dir)
    models_dir = resolve_active_models_dir(supplied_models_dir)
    if models_dir != supplied_models_dir:
        print(f"Resolved active champion directory: {models_dir}")

    validate_runtime_artifacts(
        ArtifactContract(
            models_dir=models_dir,
            transformer_required=args.transformer_required,
            allow_legacy_artifacts=args.allow_legacy_artifacts,
        )
    )

    if args.projection_csv:
        validate_projection_csv(Path(args.projection_csv))

    print("✅ Contracts passed")


if __name__ == "__main__":
    main()
