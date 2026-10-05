# Download official FineWeb shard prefixes + the official SP1024 tokenizer
# through the hf-mirror.com endpoint, then build the micro dataset.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/download_micro_data.ps1

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$dl = Join-Path $repo "downloads"
New-Item -ItemType Directory -Force -Path $dl | Out-Null

$base = "https://hf-mirror.com/datasets/willdepueoai/parameter-golf/resolve/main"
$files = @{
    "manifest.json"                = "$base/datasets/manifest.json"
    "fineweb_1024_bpe.model"       = "$base/datasets/tokenizers/fineweb_1024_bpe.model"
    "train_head.bin"               = "$base/datasets/datasets/fineweb10B_sp1024/train_000000.bin"
    "val_head.bin"                 = "$base/datasets/datasets/fineweb10B_sp1024/val_000000.bin"
}

# Range requests: header + 8M train tokens / 2M val tokens (uint16)
foreach ($name in $files.Keys) {
    $out = Join-Path $dl $name
    if ($name -eq "train_head.bin") {
        curl.exe -L -r 0-16001023 -o $out $files[$name]
    } elseif ($name -eq "val_head.bin") {
        curl.exe -L -r 0-4001023 -o $out $files[$name]
    } else {
        curl.exe -L -o $out $files[$name]
    }
}

python (Join-Path $repo "prepare_local_data.py") --prefix-dir $dl --tokenizer (Join-Path $dl "fineweb_1024_bpe.model")
Write-Output "micro data ready under $repo\data"
