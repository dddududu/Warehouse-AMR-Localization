$ErrorActionPreference = "Stop"

$jobs = @(
    "train_jun15_ccw_run2",
    "train_jun15_cw_run2",
    "val_jun23_ccw_run2",
    "val_jun23_cw_run2"
)

$outputDirectory = "outputs/semantic_stability_20260731/experiment3"
New-Item -ItemType Directory -Force $outputDirectory | Out-Null

foreach ($job in $jobs) {
    $configPath = "experiments/semantic_stability_20260731/experiment3_configs/$job`_fine.yaml"
    $logPath = "$outputDirectory/$job.collection.log"
    & python -m localization.deep_fine_localizer `
        --config $configPath `
        --frame-stride 8 `
        --save-every 25 `
        --no-resume *>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "Counterfactual collection failed for $job."
    }
}
