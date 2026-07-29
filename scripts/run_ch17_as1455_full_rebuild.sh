#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-}"
case "$mode" in
  independent-folds|backtest|backtest-only)
    exec bash scripts/run_ch17_as1455_backtest_only.sh
    ;;
  r05-addon-comparison|r05-addon-folds|r05-addon-nested)
    exec bash scripts/run_as1455_r05_addon_fold_comparison.sh
    ;;
  r05-addon-first3-ensemble|r05-first3-ensemble|first3-ensemble-nested)
    exec bash scripts/run_as1455_r05_addon_first3_ensemble_nested.sh
    ;;
  r05-addon-first3-global-forward|r05-first3-global-forward|first3-ensemble-global-forward)
    exec bash scripts/run_as1455_r05_addon_first3_ensemble_global_forward.sh
    ;;
  r05-addon-first3-global-forward-refresh|r05-first3-global-forward-refresh|first3-ensemble-global-forward-refresh)
    exec bash scripts/run_as1455_r05_addon_first3_global_forward_refresh.sh
    ;;
  r05-addon-first3-global-forward-plots|r05-first3-global-forward-plots|first3-ensemble-global-forward-plots)
    exec bash scripts/plot_as1455_r05_addon_first3_global_forward.sh
    ;;
  r05-addon-best-global-forward|r05-best-global-forward|best-model-global-forward)
    exec bash scripts/run_as1455_r05_addon_best_model_global_forward.sh
    ;;
  r05-addon-best-global-forward-refresh|r05-best-global-forward-refresh|best-model-global-forward-refresh)
    exec bash scripts/run_as1455_r05_addon_best_model_global_forward_refresh.sh
    ;;
  r05-addon-best-global-forward-plots|r05-best-global-forward-plots|best-model-global-forward-plots)
    exec bash scripts/plot_as1455_r05_addon_best_model_global_forward.sh
    ;;
  fixed-signal-matrix|requested-fixed-signal-matrix|seven-global-experiments)
    exec bash scripts/run_as1455_requested_global_fixed_signal_matrix.sh
    ;;
  r05-all5-global-forward)
    export TARGET_COL=r05_fwd REBALANCE_EVERY=5 SIGNAL_KIND=all5
    export OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/requested_v1/r05_all5_reb5_fold0_5_forward}"
    exec bash scripts/run_as1455_global_fixed_signal_experiment.sh
    ;;
  r01-all5-global-forward)
    export TARGET_COL=r01_fwd REBALANCE_EVERY=1 SIGNAL_KIND=all5
    export OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/requested_v1/r01_all5_reb1_fold0_5_forward}"
    exec bash scripts/run_as1455_global_fixed_signal_experiment.sh
    ;;
  r01-first3-global-forward)
    export TARGET_COL=r01_fwd REBALANCE_EVERY=1 SIGNAL_KIND=first3
    export OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/requested_v1/r01_first3_reb1_fold0_5_forward}"
    exec bash scripts/run_as1455_global_fixed_signal_experiment.sh
    ;;
  r01-best-global-forward)
    export TARGET_COL=r01_fwd REBALANCE_EVERY=1 SIGNAL_KIND=best
    export OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/requested_v1/r01_best_reb1_fold0_5_forward}"
    exec bash scripts/run_as1455_global_fixed_signal_experiment.sh
    ;;
  r21-all5-global-forward)
    export TARGET_COL=r21_fwd REBALANCE_EVERY=21 SIGNAL_KIND=all5
    export OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/requested_v1/r21_all5_reb21_fold0_5_forward}"
    exec bash scripts/run_as1455_global_fixed_signal_experiment.sh
    ;;
  r21-first3-global-forward)
    export TARGET_COL=r21_fwd REBALANCE_EVERY=21 SIGNAL_KIND=first3
    export OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/requested_v1/r21_first3_reb21_fold0_5_forward}"
    exec bash scripts/run_as1455_global_fixed_signal_experiment.sh
    ;;
  r21-best-global-forward)
    export TARGET_COL=r21_fwd REBALANCE_EVERY=21 SIGNAL_KIND=best
    export OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/requested_v1/r21_best_reb21_fold0_5_forward}"
    exec bash scripts/run_as1455_global_fixed_signal_experiment.sh
    ;;
  r05-addon-plots|r05-addon-plot-only)
    exec bash scripts/plot_as1455_nested_fold_results.sh
    ;;
  existing-results|plots)
    exec bash scripts/run_ch17_as1455_existing_results.sh
    ;;
  all|preflight|data|selfcheck|training|historical|forward|audit|status|"")
    echo "[BLOCKED] The unrestricted full-rebuild entry is disabled on this branch." >&2
    echo "Run the requested seven fixed-signal global experiments:" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh fixed-signal-matrix" >&2
    echo "Individual new experiment aliases:" >&2
    echo "  r05-all5-global-forward" >&2
    echo "  r01-all5-global-forward | r01-first3-global-forward | r01-best-global-forward" >&2
    echo "  r21-all5-global-forward | r21-first3-global-forward | r21-best-global-forward" >&2
    echo "Existing corrected/nested controls remain available:" >&2
    echo "  r05-addon-comparison" >&2
    echo "  r05-addon-first3-ensemble" >&2
    echo "  r05-addon-first3-global-forward" >&2
    echo "  r05-addon-best-global-forward" >&2
    exit 2
    ;;
  *)
    echo "[ERROR] unsupported mode: $mode" >&2
    echo "Usage:" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh fixed-signal-matrix" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh r05-all5-global-forward" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh r01-{all5,first3,best}-global-forward" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh r21-{all5,first3,best}-global-forward" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh r05-addon-comparison" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh r05-addon-first3-global-forward" >&2
    echo "  bash scripts/run_ch17_as1455_full_rebuild.sh r05-addon-best-global-forward" >&2
    exit 2
    ;;
esac
