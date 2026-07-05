#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[INFO] This deletes known bad DERIVED AS1455 artifacts only."
echo "[INFO] It keeps raw/intermediate cache dirs:"
echo "       saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache"
echo "       saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache"
echo "       saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache"

CH12="saved_data/ashare_ml4t/ch12_as1455"

rm -f \
  "${CH12}/model_data_as1455.h5" \
  "${CH12}/model_data_as1455.h5.bak_before_extend_"* \
  "${CH12}/as1455_ohlcv_raw.h5" \
  "${CH12}/as1455_ohlcv_adj.h5" \
  "${CH12}/as1455_execution_metadata.h5" \
  "${CH12}/model_data_contract.json" \
  "${CH12}/model_data_contract_validation.json" 2>/dev/null || true

rm -rf "${CH12}/reports" 2>/dev/null || true

# Known bad weekly retrain/backtest outputs from raw/sectorfix extension attempts.
rm -rf \
  saved_data/ashare_ml4t/ch17_as1455_weekly_retrain_empty_20260516_to_2026-06-26_after_hole_repair* \
  saved_data/ashare_ml4t/ch17_as1455_weekly_retrain_empty_20260516_to_2026-06-26_sectorfix* \
  saved_data/ashare_ml4t/ch17_as1455_weekly_retrain_empty_20260516_to_20260626_after_hole_repair* \
  saved_data/ashare_ml4t/ch17_as1455_weekly_retrain_empty_20260516_to_20260626_sectorfix* 2>/dev/null || true

echo "[DONE] bad derived artifacts removed"
