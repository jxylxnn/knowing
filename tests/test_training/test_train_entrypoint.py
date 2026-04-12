import ast
from pathlib import Path


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _is_inside_named_function(node: ast.AST, parents: dict[ast.AST, ast.AST], function_name: str) -> bool:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.FunctionDef) and current.name == function_name:
            return True
        current = parents.get(current)
    return False


def test_train_entrypoint_uses_compat_feature_engineer_helper():
    source = Path("train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    parents = _parent_map(tree)

    compat_calls = []
    direct_ctor_calls = []
    bare_ctor_calls = []

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

        if (
            name == "FeatureEngineer"
            and not node.args
            and not keywords
            and not _is_inside_named_function(node, parents, "build_feature_engineer")
        ):
            bare_ctor_calls.append(node)

    assert compat_calls, "train.py should construct FeatureEngineer via build_feature_engineer(...)"
    assert not direct_ctor_calls, (
        "train.py should not pass disable_groups/disable_columns directly into "
        "FeatureEngineer(...)"
    )
    assert not bare_ctor_calls, "train.py should not instantiate FeatureEngineer() directly in Step 2"
