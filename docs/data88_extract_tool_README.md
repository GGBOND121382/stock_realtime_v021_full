# data88 Extract Tool

This small tool extracts selected stock/ETF folders from data88 daily `.7z`
archives and writes one `.zip` package per symbol/date.

## Install

```powershell
python -m pip install -r requirements/data88_extract_tool_requirements.txt
```

## List Matched Daily Archives

```powershell
python data_collection/batch_extract_data88_selected.py --archive-dir D:\PurchasedData --pattern *.7z --list-archives
```

The archive scan is recursive by default, so subdirectories such as
`D:\PurchasedData\202602\20260203.7z` are included.

## Batch Extract

```powershell
python data_collection/batch_extract_data88_selected.py --archive-dir D:\PurchasedData --pattern *.7z --symbols-file selected_watchlist.txt --overwrite
```

Output is written by default under `saved_data/data88_selected`:

```text
saved_data/data88_selected/YYYYMMDD/_zip/YYYYMMDD_SYMBOL.zip
```

Each zip keeps the symbol folder inside, for example:

```text
002714.SZ/行情.csv
002714.SZ/逐笔委托.csv
002714.SZ/逐笔成交.csv
```

## Extract One Archive

```powershell
python data_collection/extract_data88_selected.py extract-zip --archive D:\PurchasedData\20260331.7z --symbols 002714,601899.SH --backend py7zr --overwrite
```

The default `py7zr` path converts from the source `.7z` to `.zip` through
memory and does not leave persistent intermediate extracted folders.
