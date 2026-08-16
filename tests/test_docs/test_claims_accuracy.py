"""Narrow documentation-accuracy regression tests.

These checks exist to stop README.md / AGENTS.md from drifting back into
claims that the code, CLI, or deployed artifacts do not support. They are
deliberately narrow: they pin wording, not test counts or group counts that
legitimately change over time.
"""

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
README_PATH = PROJECT_ROOT / "README.md"
AGENTS_PATH = PROJECT_ROOT / "AGENTS.md"

# Exact unconditional phrase that must never reappear in either doc.
UNCONDITIONAL_STACK_PHRASE = "Active model stack: CatBoost + Transformer"


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def agents_text() -> str:
    return AGENTS_PATH.read_text(encoding="utf-8")


def test_docs_do_not_claim_260_plus_tests(readme_text: str, agents_text: str) -> None:
    assert "260+" not in readme_text
    assert "260+" not in agents_text


def test_docs_do_not_claim_an_unconditional_active_stack(
    readme_text: str, agents_text: str
) -> None:
    # The stack is artifact-dependent: supported/default/deployed must be
    # distinguished instead of asserting an always-active CatBoost+Transformer.
    assert UNCONDITIONAL_STACK_PHRASE not in readme_text
    assert UNCONDITIONAL_STACK_PHRASE not in agents_text


def test_readme_does_not_call_disabled_optimization_continuous(readme_text: str) -> None:
    # Self-optimization is a manual, disabled-by-default capability. It must
    # not be described as continuously retuning weights.
    lowered = readme_text.lower()
    assert "continuously retunes" not in lowered
    assert "continuously retune" not in lowered


def test_readme_optimizer_examples_use_real_cli_flags(
    readme_text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every ``optimize_weights.py`` example in the README must parse."""
    import optimize_weights  # local import: CLI parser is the thing under test

    examples = [
        line.strip().removeprefix("python optimize_weights.py").split()
        for line in readme_text.splitlines()
        if line.strip().startswith("python optimize_weights.py")
    ]
    assert examples, "no optimize_weights.py examples found in README"
    for argv in examples:
        # parse_args() reads sys.argv; it takes no positional arguments.
        monkeypatch.setattr(sys, "argv", ["optimize_weights.py", *argv])
        try:
            optimize_weights.parse_args()
        except SystemExit as exc:  # argparse exits non-zero on unknown flags
            pytest.fail(
                "README optimizer example uses flags rejected by parse_args(): "
                f"{' '.join(argv)!r} (exit code {exc.code})"
            )


def test_readme_does_not_promise_a_fixed_13_dim_search(readme_text: str) -> None:
    # The search layout is derived from the loaded components: 12 parameters
    # for a six-target Transformer bundle, 13 only when an MAE companion blend
    # is also active. A fixed "13-dim" promise would be wrong for CatBoost-only
    # artifacts.
    assert "13-dim" not in readme_text
    assert "13-dimensional" not in readme_text
