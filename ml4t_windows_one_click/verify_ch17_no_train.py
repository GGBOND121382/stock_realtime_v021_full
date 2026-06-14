import ast
import json
import re
import tempfile
from ast import literal_eval as make_tuple
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
REPO = ROOT / "machine-learning-for-trading"
NB12 = REPO / "12_gradient_boosting_machines" / "04_preparing_the_model_data.ipynb"
NB17 = REPO / "17_deep_learning" / "04_optimizing_a_NN_architecture_for_trading.ipynb"
NB17_BACKTEST = REPO / "17_deep_learning" / "05_backtesting_with_zipline.ipynb"


def read_notebook_code(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            cells.append((i, src))
    return cells


def strip_magics(src):
    return "\n".join(
        line for line in src.splitlines()
        if not line.lstrip().startswith(("%", "!", "?"))
    )


def assert_code_cells_parse():
    for path in [NB12, NB17, NB17_BACKTEST]:
        for cell_no, src in read_notebook_code(path):
            code = strip_magics(src)
            if code.strip():
                ast.parse(code, filename=f"{path}:{cell_no}")


def assert_no_known_bad_patterns():
    bad_patterns = [
        "sys.path[0]",
        "pd.np",
        "null_counts",
        "sort_index(1)",
        "pd.Int64Index",
        ".ix[",
        ".as_matrix(",
        ".iteritems(",
        "pd.datetime",
        "error_bad_lines",
        "squeeze=",
        "mangle_dupe_cols",
        "line_terminator",
        "f'ckpt_{fold}_{epoch}'",
    ]
    for path in [NB12, NB17, NB17_BACKTEST]:
        text = path.read_text(encoding="utf-8")
        for pattern in bad_patterns:
            if pattern in text:
                raise AssertionError(f"{path} still contains {pattern!r}")
        if re.search(r"rolling\([^\)]*level\s*=", text, flags=re.S):
            raise AssertionError(f"{path} still contains rolling(..., level=...)")


def extract_function(path, name):
    for _, src in read_notebook_code(path):
        code = strip_magics(src)
        if f"def {name}(" in code:
            module = ast.parse(code)
            keep = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name]
            if keep:
                return ast.unparse(keep[0])
    raise AssertionError(f"Function {name} not found in {path}")


def test_run_ols_dynamic():
    src = extract_function(NB17, "run_ols")
    ns = {"pd": pd, "sm": sm, "params": ["dense_layers", "dropout", "batch_size"]}
    exec(src, ns)

    dates = pd.date_range("2020-01-01", periods=8)
    rows = []
    for dense_layers in ["16-8", "32-16"]:
        for dropout in [0.0, 0.1]:
            for batch_size in [64, 256]:
                row = {
                    "dense_layers": dense_layers,
                    "dropout": dropout,
                    "batch_size": batch_size,
                }
                for epoch in range(3):
                    row[epoch] = 0.01 * (epoch + 1) + 0.001 * len(rows)
                rows.append(row)
    ic = pd.DataFrame(rows, index=dates.repeat(len(rows) // len(dates)))
    result = ns["run_ols"](ic.copy())
    if not np.isfinite(result.params).all():
        raise AssertionError("run_ols returned non-finite params")


class FakeCV:
    def split(self, X):
        n = len(X)
        yield np.arange(0, n // 2), np.arange(n // 2, n)


class FakeStatus:
    def expect_partial(self):
        return None


class FakeModel:
    def load_weights(self, path):
        if not str(path).endswith(".weights.h5"):
            raise AssertionError(f"unexpected checkpoint path: {path}")
        return FakeStatus()

    def predict(self, x):
        return np.linspace(0.0, 1.0, len(x)).reshape(-1, 1)


def test_generate_predictions_dynamic():
    src = extract_function(NB17, "generate_predictions")
    idx = pd.IndexSlice

    dates = pd.date_range("2016-01-01", periods=8)
    symbols = ["A", "B"]
    index = pd.MultiIndex.from_product([symbols, dates], names=["symbol", "date"])
    fake_data = pd.DataFrame(
        {
            "feature_1": np.arange(len(index), dtype=float),
            "feature_2": np.arange(len(index), dtype=float) / 10,
            "r01_fwd": np.sin(np.arange(len(index))),
            "r05_fwd": np.cos(np.arange(len(index))),
        },
        index=index,
    )

    original_read_hdf = pd.read_hdf
    pd.read_hdf = lambda *args, **kwargs: fake_data.copy()
    try:
        ns = {
            "pd": pd,
            "np": np,
            "idx": idx,
            "cv": FakeCV(),
            "StandardScaler": StandardScaler,
            "get_train_valid_data": lambda X, y, train_idx, test_idx: (
                X.iloc[train_idx],
                y.iloc[train_idx],
                X.iloc[test_idx],
                y.iloc[test_idx],
            ),
            "make_model": lambda *args, **kwargs: FakeModel(),
            "make_tuple": make_tuple,
            "checkpoint_path": Path(tempfile.mkdtemp()),
        }
        exec(src, ns)
        out = ns["generate_predictions"]("(16, 8)", "tanh", 0.1, 64, 3)
        if not isinstance(out, pd.Series):
            raise AssertionError("generate_predictions did not return a Series")
        if out.empty or not np.isfinite(out.to_numpy()).all():
            raise AssertionError("generate_predictions returned invalid values")
    finally:
        pd.read_hdf = original_read_hdf


def test_bad_expect_partial_repair_regex():
    bad = """        status = model.load_weights((checkpoint_dir / f'ckpt_{fold}_{epoch}.weights.h5').as_posix())
        if hasattr(status, 'expect_partial'):
            if hasattr(status, 'expect_partial'):
            status.expect_partial()
        predictions.append(pd.Series(model.predict(x_val).squeeze(), index=y_val.index))
"""
    fixed = re.sub(
        r"if hasattr\(status, 'expect_partial'\):\n\s+if hasattr\(status, 'expect_partial'\):\n\s*status\.expect_partial\(\)",
        "if hasattr(status, 'expect_partial'):\n            status.expect_partial()",
        bad,
    )
    ast.parse("def f():\n" + fixed)


def main():
    assert_code_cells_parse()
    assert_no_known_bad_patterns()
    test_run_ols_dynamic()
    test_generate_predictions_dynamic()
    test_bad_expect_partial_repair_regex()
    print("CH17 no-train verification passed")


if __name__ == "__main__":
    main()
