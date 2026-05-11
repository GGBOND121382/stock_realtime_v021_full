# ths_sector_hedged_cache_v2_patch

本补丁是在 `ths_sector_hedged_cache_patch` 基础上的小修。

## 修复点

上一版 hedged request 已经能工作，但实现方式仍有瑕疵：

```text
第 0 个请求已经成功时，ThreadPoolExecutor 仍可能等待延迟的第 1 个任务结束；
也可能无谓发出第二个 THS 请求。
```

这版改成真正的按需错峰：

```text
1. 先启动第 0 个 subprocess；
2. 等 hedge_delay_seconds；
3. 如果第 0 个已经成功，立即返回，不启动第 1 个；
4. 如果第 0 个还没成功或失败，再启动第 1 个；
5. 谁先成功就返回，并终止剩余 subprocess。
```

## 默认参数

```text
sector_request_timeout_seconds = 5.0
sector_hedge_workers = 2
sector_hedge_delay_seconds = 1.5
```

## 覆盖方式

```bash
unzip -o ths_sector_hedged_cache_v2_patch.zip -d .

python3 -m py_compile \
  data_collection/collect_realtime_context.py \
  tests/test_ths_sector_hedged_cache.py
```

## 测试

```bash
source .venv/bin/activate
python3 tests/test_ths_sector_hedged_cache.py --rounds 3 --timeout 5 --hedge-workers 2 --hedge-delay 1.5
cat debug_ths_sector_hedged/summary.json
```

重点看每轮 meta：

```json
"status": "ok",
"winner": 0,
"launched_workers": 1
```

如果 `winner=0` 且 `launched_workers=1`，说明第一个请求成功后没有再发第二个请求。  
如果第一个请求卡住，`launched_workers` 可能为 2，这是正常的补救行为。
