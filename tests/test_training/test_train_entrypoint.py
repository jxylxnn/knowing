import ast
from pathlib import Path


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def test_train_entrypoint_uses_compat_feature_engineer_helper():
    source = Path("train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    compat_calls = []
    direct_ctor_calls = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        keywords = {kw.arg for kw in node.keywords if kw.arg is not None}
        name = _call_name(node)

        if name == "build_feature_engineer" and {
            "disable_groups",
            "disable_columns",
        }.issubset(keywords):
            compat_calls.append(node)

        if name == "FeatureEngineer" and (
            "disable_groups" in keywords or "disable_columns" in keywords
        ):
            direct_ctor_calls.append(node)

    assert compat_calls, "train.py should construct FeatureEngineer via build_feature_engineer(...)"
    assert not direct_ctor_calls, (
        "train.py should not pass disable_groups/disable_columns directly into "
        "FeatureEngineer(...)"
    )
