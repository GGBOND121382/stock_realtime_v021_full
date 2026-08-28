from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "android_executor_autojs6/main.js"
R01 = ROOT / "android_executor_autojs6/fetch_submit_test.js"


def test_r01_executor_keeps_main_execution_safety_semantics() -> None:
    main = MAIN.read_text(encoding="utf-8")
    r01 = R01.read_text(encoding="utf-8")

    shared_contract = [
        'require("./lib/safety.js")',
        'require("./lib/ledger.js")',
        'require("./lib/plan_client.js")',
        'require("./lib/ths_adapter.js")',
        'require("./lib/retry_queue.js")',
        'cfg.mode !== "dry_run" && cfg.mode !== "live"',
        "ledger.isTerminal(order.signal_id)",
        "ledger.markStarted(order)",
        'config.mode === "live" ? ths.submit(order, config) : ths.preview(order, config)',
        "ledger.markResult(order, result, config.mode !== \"live\")",
        "retryQueue.remove(order.signal_id)",
        "ledger.markManualRequired(order, e)",
        "ledger.markFailed(order, e)",
        "ths.recoverToTradingPage(order.side, config)",
        "e.ambiguous === true || e.fatal_ui_state === true",
        "!ledger.isTerminal(o.signal_id)",
        "config.require_manual_confirm !== false",
        "retryQueue.path()",
    ]
    for needle in shared_contract:
        assert needle in main
        assert needle in r01


def test_r01_executor_differs_only_at_plan_selection_boundary() -> None:
    r01 = R01.read_text(encoding="utf-8")

    assert 'DEFAULT_TEST_EXPERIMENT = "r01_best_reb1_fold0_5_forward"' in r01
    assert "cfg.production_experiment = experiment" in r01
    assert "client.fetchLatest(config, config.production_experiment)" in r01
    assert "raw.temporary_test_batch === true" in r01
    assert "allow_batch_live_test must be true" not in r01
    assert "fetch_submit_test_result.json" not in r01
