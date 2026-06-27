# Build + run each dockerised learned baseline SEQUENTIALLY (one heavy Docker op at a time, to avoid the
# WSL2 memory pressure that drops the buildkit daemon), then score everything against GT.
#   powershell -File baselines_ext\run_all.ps1                       # all methods
#   powershell -File baselines_ext\run_all.ps1 neuralpull capudf     # a subset
param([string[]]$Methods = @("neuralpull", "capudf", "sap", "poco", "nksr", "convonet"))
$repo = "C:\work\Points_as_supertoroids"

# 1. export the comparison clouds once (CPU)
if (-not (Test-Path "$repo\baselines_ext\clouds\cube.npz")) {
    docker run --rm -v "${repo}:/workspace" -w /workspace -e PYTHONPATH=/workspace `
        waveshape-compare python baselines_ext/export_clouds.py
}

foreach ($m in $Methods) {
    Write-Output "===================== $m : BUILD ====================="
    docker build -t "wsn-bl-$m" "$repo\baselines_ext\$m"
    if (-not $?) { Write-Output "[$m] BUILD FAILED -- skipping"; continue }
    New-Item -ItemType Directory -Force "$repo\baselines_ext\out\$m" | Out-Null
    Write-Output "===================== $m : RUN ======================="
    docker run --rm --gpus all `
        -v "$repo\baselines_ext\clouds:/in" `
        -v "$repo\baselines_ext\out\$m:/out" "wsn-bl-$m" /in /out
    if (-not $?) { Write-Output "[$m] RUN FAILED" }
}

Write-Output "===================== SCORE ==========================="
docker run --rm -v "${repo}:/workspace" -w /workspace -e PYTHONPATH=/workspace `
    waveshape-compare python baselines_ext/score.py
Write-Output "done -> baselines_ext/learned_metrics.json"
