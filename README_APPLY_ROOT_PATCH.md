# Root-level patch package

This zip is intentionally root-relative. Its top-level entries are:

- scripts/
- model_saving/
- patches/
- tools/

Apply from project root:

```bash
unzip -o update_ranked_models_patch_v4_root.zip -d /root/stock_realtime_v021_full
cd /root/stock_realtime_v021_full
bash scripts/apply_ranked_models_data_patch.sh
bash scripts/apply_realtime_5m_bar_patch.sh
```

Verify zip structure before applying:

```bash
unzip -l update_ranked_models_patch_v4_root.zip | head -40
```

There should be no leading `update_ranked_models_patch_v4_root/` directory in paths.
