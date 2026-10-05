# End-to-end CPU smoke test (Windows): data prep -> one tiny run -> pack check.
$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "download_micro_data.ps1")
python (Join-Path $repo "scripts\run_experiments.py") --quick
python (Join-Path $repo "scripts\analyze.py")
