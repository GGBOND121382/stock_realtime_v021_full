import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT / "machine-learning-for-trading"
DEFAULT_NOTEBOOKS = [
    REPO / "12_gradient_boosting_machines" / "04_preparing_the_model_data.ipynb",
    REPO / "17_deep_learning" / "04_optimizing_a_NN_architecture_for_trading.ipynb",
    REPO / "17_deep_learning" / "05_backtesting_with_zipline.ipynb",
]


def patch_source(src: str) -> str:
    patched = src

    # Repair a previously non-idempotent patch that could create nested if blocks
    # with invalid indentation.
    patched = re.sub(
        r"if hasattr\(status, 'expect_partial'\):\n\s+if hasattr\(status, 'expect_partial'\):\n\s*status\.expect_partial\(\)",
        "if hasattr(status, 'expect_partial'):\n            status.expect_partial()",
        patched,
    )

    patched = patched.replace("null_counts=True", "show_counts=True")
    patched = patched.replace(".sort_index(1)", ".sort_index(axis=1)")
    patched = patched.replace("pd.np.arange", "np.arange")
    patched = patched.replace(
        "sys.path.insert(1, os.path.join(sys.path[0], '..'))",
        "sys.path.insert(1, str(Path.cwd().resolve().parent))",
    )
    patched = patched.replace(
        "prices.groupby(level='symbol').close.apply(RSI)",
        "prices.groupby(level='symbol', group_keys=False).close.apply(RSI)",
    )
    patched = patched.replace(
        ".groupby(level='symbol')\n                      .close\n                      .apply(compute_bb)",
        ".groupby(level='symbol', group_keys=False)\n                      .close\n                      .apply(compute_bb)",
    )
    patched = patched.replace(
        "prices.groupby(level='symbol').close.apply(talib.PPO)",
        "prices.groupby(level='symbol', group_keys=False).close.apply(talib.PPO)",
    )
    patched = patched.replace(
        ".groupby(level='date')\n                             .apply(lambda x: pd.qcut",
        ".groupby(level='date', group_keys=False)\n                             .apply(lambda x: pd.qcut",
    )
    patched = patched.replace(
        "preds.groupby(level='date').apply(lambda x: spearmanr(x.actual, x[epoch])[0])",
        "preds.groupby(level='date', group_keys=False).apply(lambda x: spearmanr(x.actual, x[epoch])[0])",
    )
    patched = patched.replace(
        "model_data.columns = [s.split('_')[-1] for s in model_data.columns]\n    model = sm.OLS",
        "model_data.columns = [s.split('_')[-1] for s in model_data.columns]\n    model_data = model_data.apply(pd.to_numeric, errors='coerce').astype(float)\n    model = sm.OLS",
    )
    patched = patched.replace("f'ckpt_{fold}_{epoch}'", "f'ckpt_{fold}_{epoch}.weights.h5'")
    patched = patched.replace(
        "        status.expect_partial()",
        "        if hasattr(status, 'expect_partial'):\n            status.expect_partial()",
    )
    patched = patched.replace(
        "pd.Int64Index([asset.sid for asset in assets])",
        "pd.Index([asset.sid for asset in assets], dtype='int64')",
    )
    patched = patched.replace(
        "ic = []\nscaler = StandardScaler()\nfor params in param_grid:",
        "if (results_path / 'scores.h5').exists():\n    print('Skipping NN CV training because results/scores.h5 already exists.')\n    param_grid = []\n\nic = []\nscaler = StandardScaler()\nfor params in param_grid:",
    )

    # Collapse again in case the guarded expect_partial replacement was applied
    # to an already-guarded block.
    patched = re.sub(
        r"if hasattr\(status, 'expect_partial'\):\n\s+if hasattr\(status, 'expect_partial'\):\n\s*status\.expect_partial\(\)",
        "if hasattr(status, 'expect_partial'):\n            status.expect_partial()",
        patched,
    )
    return patched


def patch_notebook(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        patched = patch_source(src)
        if patched != src:
            cell["source"] = patched.splitlines(keepends=True)
            changed = True
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
    if changed:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_NOTEBOOKS
    for path in paths:
        if not path.exists():
            raise SystemExit(f"missing notebook: {path}")
        changed = patch_notebook(path)
        print(("patched" if changed else "ok") + f": {path}")


if __name__ == "__main__":
    main()
