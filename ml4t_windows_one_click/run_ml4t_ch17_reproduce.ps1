param(
    [string]$WorkDir = "",
    [string]$RepoDir = "",
    [string]$Python = "",
    [switch]$ForceAssets,
    [switch]$ForceChapter12,
    [switch]$ForceTraining,
    [switch]$SkipBacktest,
    [switch]$InstallBacktestDeps
)

$ErrorActionPreference = "Stop"

function Split-PythonCommand {
    param([string]$Command)
    if ($Command -ne "") {
        return ,($Command -split '\s+')
    }

    $scriptDir = Split-Path -Parent $MyInvocation.ScriptName
    $projectVenv = Join-Path (Split-Path -Parent $scriptDir) ".venv\Scripts\python.exe"
    if (Test-Path $projectVenv) {
        return ,@($projectVenv)
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return ,@("py", "-3")
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return ,@("python")
    }
    throw "Python not found. Pass -Python path\to\python.exe or ensure py/python is on PATH."
}

function Ensure-PythonPackages {
    param(
        [string]$PythonExe,
        [object[]]$PythonPrefix,
        [string[]]$Imports,
        [string[]]$Packages,
        [string]$Label
    )

    $importCode = @"
import importlib.util
import sys
missing = [m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]
print(chr(44).join(missing))
raise SystemExit(1 if missing else 0)
"@

    $check = & $PythonExe @PythonPrefix -c $importCode @Imports
    if ($LASTEXITCODE -eq 0) {
        Write-Host "$Label dependencies ok"
        return
    }

    if ($Packages.Count -eq 0) {
        $missing = ($check | Select-Object -Last 1)
        throw "$Label dependencies missing: $missing"
    }

    $missing = ($check | Select-Object -Last 1)
    Write-Host "$Label dependencies missing: $missing"
    Write-Host "Installing packages: $($Packages -join ' ')"
    & $PythonExe @PythonPrefix -m pip install @Packages
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed for $Label dependencies."
    }

    & $PythonExe @PythonPrefix -c $importCode @Imports | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "$Label dependency check still failed after install."
    }
    Write-Host "$Label dependencies installed"
}

function Test-HdfKey {
    param(
        [string]$PythonExe,
        [object[]]$PythonPrefix,
        [string]$Path,
        [string]$Key
    )

    if (-not (Test-Path $Path)) {
        return $false
    }
    $code = @"
import sys
import pandas as pd
path, key = sys.argv[1], sys.argv[2]
try:
    with pd.HDFStore(path) as store:
        ok = key in store.keys() or ('/' + key.lstrip('/')) in store.keys()
    raise SystemExit(0 if ok else 1)
except Exception:
    raise SystemExit(1)
"@
    & $PythonExe @PythonPrefix -c $code $Path $Key | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Invoke-Notebook {
    param(
        [string]$PythonExe,
        [object[]]$PythonPrefix,
        [string]$NotebookPath,
        [string]$OutputName
    )

    $resolved = (Resolve-Path $NotebookPath).Path
    $notebookDir = Split-Path -Parent $resolved
    $notebookName = Split-Path -Leaf $resolved
    Write-Host "Executing notebook: $resolved"
    Push-Location $notebookDir
    try {
        & $PythonExe @PythonPrefix -m jupyter nbconvert `
            --to notebook `
            --execute $notebookName `
            --output $OutputName `
            --ExecutePreprocessor.timeout=-1
        if ($LASTEXITCODE -ne 0) {
            throw "Notebook failed: $resolved"
        }
    } finally {
        Pop-Location
    }
}

function Patch-NotebookCompatibility {
    param([string]$Path)

    $raw = Get-Content -LiteralPath $Path -Raw
    $patched = $raw
    $patched = $patched.Replace("null_counts=True", "show_counts=True")
    $patched = $patched.Replace(".sort_index(1)", ".sort_index(axis=1)")
    $patched = $patched.Replace("pd.np.arange", "np.arange")
    $patched = $patched.Replace("sys.path.insert(1, os.path.join(sys.path[0], '..'))", "sys.path.insert(1, str(Path.cwd().resolve().parent))")
    $patched = $patched.Replace("prices.groupby(level='symbol').close.apply(RSI)", "prices.groupby(level='symbol', group_keys=False).close.apply(RSI)")
    $patched = $patched.Replace(".groupby(level='symbol')\n                      .close\n                      .apply(compute_bb)", ".groupby(level='symbol', group_keys=False)\n                      .close\n                      .apply(compute_bb)")
    $patched = $patched.Replace("prices.groupby(level='symbol').close.apply(talib.PPO)", "prices.groupby(level='symbol', group_keys=False).close.apply(talib.PPO)")
    $patched = $patched.Replace(".groupby(level='date')\n                             .apply(lambda x: pd.qcut", ".groupby(level='date', group_keys=False)\n                             .apply(lambda x: pd.qcut")
    $patched = $patched.Replace("preds.groupby(level='date').apply(lambda x: spearmanr(x.actual, x[epoch])[0])", "preds.groupby(level='date', group_keys=False).apply(lambda x: spearmanr(x.actual, x[epoch])[0])")
    $patched = $patched.Replace("f'ckpt_{fold}_{epoch}'", "f'ckpt_{fold}_{epoch}.weights.h5'")
    $patched = $patched.Replace("status.expect_partial()", "if hasattr(status, 'expect_partial'):\n            status.expect_partial()")
    $patched = $patched.Replace("pd.Int64Index([asset.sid for asset in assets])", "pd.Index([asset.sid for asset in assets], dtype='int64')")
    if ($patched -ne $raw) {
        Set-Content -LiteralPath $Path -Value $patched -Encoding UTF8
        Write-Host "Patched notebook compatibility: $Path"
    }
}

function Build-AssetsFromWikiCsv {
    param(
        [string]$PythonExe,
        [object[]]$PythonPrefix,
        [string]$DataDir
    )

    $code = @"
from pathlib import Path
import pandas as pd

base = Path(r'''$DataDir''')
store_path = base / 'assets.h5'
prices_path = base / 'wiki_prices.csv'
stocks_path = base / 'wiki_stocks.csv'
us_meta_path = base / 'us_equities_meta_data.csv'

if not prices_path.exists():
    raise SystemExit(f'missing {prices_path}')
if not stocks_path.exists():
    raise SystemExit(f'missing {stocks_path}')
if not us_meta_path.exists():
    raise SystemExit(f'missing {us_meta_path}')

if store_path.exists():
    store_path.unlink()

print(f'Reading {prices_path} ...', flush=True)
prices = (pd.read_csv(
    prices_path,
    parse_dates=['date'],
    index_col=['date', 'ticker'],
).sort_index())
print(f'Writing {store_path}:/quandl/wiki/prices ...', flush=True)
with pd.HDFStore(store_path) as store:
    store.put('quandl/wiki/prices', prices)
del prices

print('Writing /quandl/wiki/stocks ...', flush=True)
wiki_stocks = pd.read_csv(stocks_path)
with pd.HDFStore(store_path) as store:
    store.put('quandl/wiki/stocks', wiki_stocks)
del wiki_stocks

print('Writing /us_equities/stocks ...', flush=True)
us_meta = pd.read_csv(us_meta_path)
with pd.HDFStore(store_path) as store:
    store.put('us_equities/stocks', us_meta.set_index('ticker'))
    print('Keys:', store.keys(), flush=True)
"@

    & $PythonExe @PythonPrefix -c $code
    if ($LASTEXITCODE -ne 0) {
        throw "assets.h5 build failed."
    }
}

Write-Host "=== ML4T Chapter 17 reproducibility runner ==="

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($WorkDir -eq "") {
    $WorkDir = $scriptDir
}
$WorkDir = (Resolve-Path $WorkDir).Path

if ($RepoDir -eq "") {
    $RepoDir = Join-Path $WorkDir "machine-learning-for-trading"
}
if (-not (Test-Path $RepoDir)) {
    throw "Repo directory not found: $RepoDir. Copy machine-learning-for-trading under $WorkDir first."
}
$RepoDir = (Resolve-Path $RepoDir).Path

$pythonArgs = Split-PythonCommand -Command $Python
$pythonExe = $pythonArgs[0]
$pythonPrefix = @()
if ($pythonArgs.Count -gt 1) {
    $pythonPrefix = $pythonArgs[1..($pythonArgs.Count - 1)]
}

Write-Host "WorkDir: $WorkDir"
Write-Host "RepoDir: $RepoDir"
Write-Host "Python : $($pythonArgs -join ' ')"

Write-Host "Checking Python dependencies..."
Ensure-PythonPackages `
    -PythonExe $pythonExe `
    -PythonPrefix $pythonPrefix `
    -Imports @("numpy", "pandas", "tables", "sklearn", "matplotlib", "seaborn", "scipy", "statsmodels", "jupyter", "nbconvert", "nbformat", "talib", "tensorflow", "tensorboard") `
    -Packages @("numpy", "pandas", "tables", "scikit-learn", "matplotlib", "seaborn", "scipy", "statsmodels", "jupyter", "nbconvert", "nbformat", "TA-Lib", "tensorflow", "tensorboard") `
    -Label "Chapter 12/17 training"

if ($InstallBacktestDeps) {
    Ensure-PythonPackages `
        -PythonExe $pythonExe `
        -PythonPrefix $pythonPrefix `
        -Imports @("zipline", "pyfolio", "alphalens", "trading_calendars", "logbook") `
        -Packages @("zipline-reloaded", "pyfolio-reloaded", "alphalens-reloaded", "trading-calendars", "logbook") `
        -Label "Zipline backtest"
}

$dataDir = Join-Path $RepoDir "data"
$assetsPath = Join-Path $dataDir "assets.h5"
$chapter12Dir = Join-Path $RepoDir "12_gradient_boosting_machines"
$chapter17Dir = Join-Path $RepoDir "17_deep_learning"
$chapter12Data = Join-Path $chapter12Dir "data.h5"
$chapter17Results = Join-Path $chapter17Dir "results"
$scoresPath = Join-Path $chapter17Results "scores.h5"
$predsPath = Join-Path $chapter17Results "test_preds.h5"

$assetsReady = (Test-HdfKey -PythonExe $pythonExe -PythonPrefix $pythonPrefix -Path $assetsPath -Key "/quandl/wiki/prices") -and
               (Test-HdfKey -PythonExe $pythonExe -PythonPrefix $pythonPrefix -Path $assetsPath -Key "/us_equities/stocks")
if ($ForceAssets -or -not $assetsReady) {
    Write-Host "Building data/assets.h5 from existing local WIKI csv files..."
    Build-AssetsFromWikiCsv -PythonExe $pythonExe -PythonPrefix $pythonPrefix -DataDir $dataDir
} else {
    Write-Host "Skipping assets.h5 build; required keys already exist."
}

$nb12 = Join-Path $chapter12Dir "04_preparing_the_model_data.ipynb"
$nb17Train = Join-Path $chapter17Dir "04_optimizing_a_NN_architecture_for_trading.ipynb"
$nb17Backtest = Join-Path $chapter17Dir "05_backtesting_with_zipline.ipynb"

Patch-NotebookCompatibility -Path $nb12
Patch-NotebookCompatibility -Path $nb17Train
Patch-NotebookCompatibility -Path $nb17Backtest

$chapter12Ready = Test-HdfKey -PythonExe $pythonExe -PythonPrefix $pythonPrefix -Path $chapter12Data -Key "/model_data"
if ($ForceChapter12 -or -not $chapter12Ready) {
    Invoke-Notebook -PythonExe $pythonExe -PythonPrefix $pythonPrefix -NotebookPath $nb12 -OutputName "04_preparing_the_model_data.executed.ipynb"
} else {
    Write-Host "Skipping Chapter 12 model data; 12_gradient_boosting_machines/data.h5::/model_data already exists."
}

$trainReady = (Test-HdfKey -PythonExe $pythonExe -PythonPrefix $pythonPrefix -Path $scoresPath -Key "/ic_by_day") -and
              (Test-HdfKey -PythonExe $pythonExe -PythonPrefix $pythonPrefix -Path $predsPath -Key "/predictions")
if ($ForceTraining -or -not $trainReady) {
    Invoke-Notebook -PythonExe $pythonExe -PythonPrefix $pythonPrefix -NotebookPath $nb17Train -OutputName "04_optimizing_a_NN_architecture_for_trading.executed.ipynb"
} else {
    Write-Host "Skipping Chapter 17 NN training; scores.h5 and test_preds.h5 already exist."
}

if ($SkipBacktest) {
    Write-Host "Skipping Zipline backtest by request."
} else {
    $ziplineCode = "import zipline, pyfolio, alphalens, trading_calendars, logbook"
    & $pythonExe @pythonPrefix -c $ziplineCode | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Zipline/pyfolio/alphalens dependencies are not importable in this Python environment. Re-run with -InstallBacktestDeps on a compatible Linux/WSL/Docker environment, or use -SkipBacktest to reproduce training/prediction only. No local market data was re-downloaded."
    }
    Invoke-Notebook -PythonExe $pythonExe -PythonPrefix $pythonPrefix -NotebookPath $nb17Backtest -OutputName "05_backtesting_with_zipline.executed.ipynb"
}

Write-Host ""
Write-Host "Done. Key outputs:"
Write-Host "  $chapter12Data"
Write-Host "  $scoresPath"
Write-Host "  $predsPath"
Write-Host "  $(Join-Path $chapter17Dir '04_optimizing_a_NN_architecture_for_trading.executed.ipynb')"
if (-not $SkipBacktest) {
    Write-Host "  $(Join-Path $chapter17Dir '05_backtesting_with_zipline.executed.ipynb')"
}
