param(
    [string]$WorkDir = "",
    [string]$RepoDir = "",
    [string]$OutDir = "",
    [string]$Python = "",
    [int]$TrainDays = 63,
    [int]$TestDays = 10,
    [int]$Lookahead = 1,
    [string]$TargetCol = "",
    [string]$Ticker = "",
    [switch]$RunNotebooks,
    [switch]$BuildYahooAssets,
    [int]$MaxTickers = 500
)

$ErrorActionPreference = "Stop"

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

Write-Host "=== ML4T Chapter 7 per-ticker RMSE runner ==="

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

if ($OutDir -eq "") {
    $OutDir = Join-Path $WorkDir "out"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path

$runnerPath = Join-Path $WorkDir "ml4t_ch7_per_ticker_rmse.py"
if (-not (Test-Path $runnerPath)) {
    throw "Python runner not found: $runnerPath"
}
$assetBuilderPath = Join-Path $WorkDir "build_yahoo_assets.py"

if ($Python -ne "") {
    $pythonArgs = $Python -split '\s+'
} elseif (Test-Path (Join-Path (Split-Path -Parent $WorkDir) ".venv\Scripts\python.exe")) {
    $pythonArgs = @((Join-Path (Split-Path -Parent $WorkDir) ".venv\Scripts\python.exe"))
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonArgs = @("py", "-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonArgs = @("python")
} else {
    throw "Python not found. Pass -Python path\to\python.exe or ensure py/python is on PATH."
}

$pythonExe = $pythonArgs[0]
$pythonPrefix = @()
if ($pythonArgs.Count -gt 1) {
    $pythonPrefix = $pythonArgs[1..($pythonArgs.Count - 1)]
}

Write-Host "WorkDir: $WorkDir"
Write-Host "RepoDir: $RepoDir"
Write-Host "OutDir : $OutDir"
Write-Host "Python : $($pythonArgs -join ' ')"

Write-Host "Checking Python dependencies..."
Ensure-PythonPackages `
    -PythonExe $pythonExe `
    -PythonPrefix $pythonPrefix `
    -Imports @("numpy", "pandas", "scipy", "sklearn", "tables", "pyarrow", "fastparquet") `
    -Packages @("numpy", "pandas", "scipy", "scikit-learn", "tables", "pyarrow", "fastparquet") `
    -Label "Core/data-reader"

if ($BuildYahooAssets) {
    if (-not (Test-Path $assetBuilderPath)) {
        throw "Yahoo asset builder not found: $assetBuilderPath"
    }
    Ensure-PythonPackages `
        -PythonExe $pythonExe `
        -PythonPrefix $pythonPrefix `
        -Imports @("yfinance") `
        -Packages @("yfinance") `
        -Label "Yahoo asset builder"
    Write-Host "Building ML4T-compatible assets.h5 from Yahoo Finance data..."
    & $pythonExe @pythonPrefix $assetBuilderPath `
        --data-dir (Join-Path $RepoDir "data") `
        --max-tickers "$MaxTickers"
    if ($LASTEXITCODE -ne 0) {
        throw "Yahoo asset build failed."
    }
}

if ($RunNotebooks) {
    Ensure-PythonPackages `
        -PythonExe $pythonExe `
        -PythonPrefix $pythonPrefix `
        -Imports @("jupyter", "nbconvert", "nbformat", "matplotlib", "seaborn", "pandas_datareader", "yfinance", "talib", "statsmodels") `
        -Packages @("jupyter", "nbconvert", "nbformat", "matplotlib", "seaborn", "pandas-datareader", "yfinance", "TA-Lib", "statsmodels") `
        -Label "Notebook"

    Write-Host "Executing book notebooks with existing Python environment..."
    Push-Location (Join-Path $RepoDir "07_linear_models")
    & $pythonExe @pythonPrefix -m jupyter nbconvert --to notebook --execute "03_preparing_the_model_data.ipynb" --output "03_preparing_the_model_data.executed.ipynb" --ExecutePreprocessor.timeout=-1
    if ($LASTEXITCODE -ne 0) { throw "Notebook failed: 03_preparing_the_model_data.ipynb" }
    & $pythonExe @pythonPrefix -m jupyter nbconvert --to notebook --execute "05_predicting_stock_returns_with_linear_regression.ipynb" --output "05_predicting_stock_returns_with_linear_regression.executed.ipynb" --ExecutePreprocessor.timeout=-1
    if ($LASTEXITCODE -ne 0) { throw "Notebook failed: 05_predicting_stock_returns_with_linear_regression.ipynb" }
    Pop-Location
}

$argsList = @(
    "--repo", $RepoDir,
    "--out", $OutDir,
    "--train-days", "$TrainDays",
    "--test-days", "$TestDays",
    "--lookahead", "$Lookahead"
)
if ($TargetCol -ne "") { $argsList += @("--target-col", $TargetCol) }
if ($Ticker -ne "") { $argsList += @("--ticker", $Ticker) }

Write-Host "Running RMSE script..."
& $pythonExe @pythonPrefix $runnerPath @argsList
if ($LASTEXITCODE -ne 0) {
    throw "RMSE script failed. See $OutDir\run_summary.txt if it was created."
}

Write-Host ""
Write-Host "Done. Outputs:"
Write-Host "  $OutDir\per_ticker_rmse.csv"
Write-Host "  $OutDir\per_date_rmse.csv"
Write-Host "  $OutDir\model_summary.csv"
Write-Host "  $OutDir\predictions.csv"
Write-Host "  $OutDir\run_summary.txt"
