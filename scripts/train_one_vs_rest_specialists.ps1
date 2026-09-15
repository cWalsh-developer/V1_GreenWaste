$ErrorActionPreference = "Stop"

$env:PYTHONPATH = "src"

$DatasetRoot = "data\processed\yolo_ikea_curated_one_vs_rest_20260820"
$Classes = @(
  "beds_mattresses",
  "chair_seating",
  "sofa",
  "storage",
  "tables_desks"
)

foreach ($ClassName in $Classes) {
  Write-Host "Training one-vs-rest specialist: $ClassName"
  .\myenv\Scripts\yolo.exe detect train `
    model=yolo11n.pt `
    data="$DatasetRoot\$ClassName\data.yaml" `
    epochs=70 `
    imgsz=960 `
    batch=8 `
    project=runs\detect `
    name="one_vs_rest_${ClassName}_20260820"
}
