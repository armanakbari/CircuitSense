#!/usr/bin/env python3

import ast
import io
import subprocess
import sys
from pathlib import Path
from tokenize import generate_tokens, untokenize, COMMENT, STRING


def get_tracked_python_files(repo_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    files = [repo_root / line.strip() for line in result.stdout.splitlines() if line.strip()]
    return files


def collect_docstring_starts(source: str) -> set[tuple[int, int]]:
    """Return set of (lineno, col_offset) positions for module/class/func docstrings."""
    starts: set[tuple[int, int]] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return starts

    # Module docstring
    if tree.body and isinstance(tree.body[0], ast.Expr):
        value = tree.body[0].value
        if isinstance(value, (ast.Str, ast.Constant)) and isinstance(getattr(value, "s", getattr(value, "value", None)), str):
            starts.add((value.lineno, value.col_offset))

    # Class and function docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, (ast.Str, ast.Constant)) and isinstance(getattr(value, "s", getattr(value, "value", None)), str):
                    starts.add((value.lineno, value.col_offset))

    return starts


def strip_python_comments(content: str) -> str:
    # Preserve shebang if present on the first line
    shebang = ""
    lines = content.splitlines(keepends=True)
    if lines and lines[0].startswith("#!"):
        shebang = lines[0]
        content_wo_shebang = "".join(lines[1:])
    else:
        content_wo_shebang = content

    docstring_starts = collect_docstring_starts(content_wo_shebang)

    tokens = list(generate_tokens(io.StringIO(content_wo_shebang).readline))
    filtered = []
    for tok in tokens:
        tok_type, tok_str, start, end, line = tok
        if tok_type == COMMENT:
            # drop comments
            continue
        if tok_type == STRING and (start[0], start[1]) in docstring_starts:
            # drop docstrings
            continue
        filtered.append(tok)

    new_content = untokenize(filtered)

    # Re-prepend shebang if it existed
    if shebang:
        # Ensure newline after shebang if not present
        if not shebang.endswith("\n"):
            shebang += "\n"
        return shebang + new_content
    return new_content


def main() -> int:
    # script path: <repo_root>/scripts/strip_comments.py
    # parents[1] is the repository root
    repo_root = Path(__file__).resolve().parents[1]
    py_files = get_tracked_python_files(repo_root)

    changed_files: list[Path] = []
    for path in py_files:
        try:
            original = path.read_text(encoding="utf-8")
        except Exception:
            continue
        stripped = strip_python_comments(original)
        if stripped != original:
            path.write_text(stripped, encoding="utf-8")
            changed_files.append(path)

    print(f"Processed {len(py_files)} Python files. Updated {len(changed_files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


