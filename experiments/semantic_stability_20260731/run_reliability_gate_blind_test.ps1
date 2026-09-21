$ErrorActionPreference = "Stop"

$jobs = @(
    "oct12_aisle_ccw_baseline",
    "oct12_aisle_ccw_reliability_gate",
    "oct12_aisle_cw_baseline",
    "oct12_aisle_cw_reliability_gate"
)
$outputDirectory = "outputs/semantic_stability_20260731/experiment3"
New-Item -ItemType Directory -Force $outputDirectory | Out-Null

foreach ($job in $jobs) {
    $configPath = "experiments/semantic_stability_20260731/experiment3_configs/blind_$job.yaml"
    $logPath = "$outputDirectory/$job.log"
    & python -m localization.deep_fine_localizer `
        --config $configPath `
        --frame-stride 8 `
        --save-every 25 `
        --no-resume *>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "Blind reliability-gate evaluation failed for $job."
    }
}
