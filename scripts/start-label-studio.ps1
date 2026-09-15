[CmdletBinding()]
param(
    [string]$DocumentRoot
)

if ([string]::IsNullOrWhiteSpace($DocumentRoot)) {
    $DocumentRoot = Join-Path -Path $PSScriptRoot -ChildPath "..\data\raw"
}

$resolvedRoot = (Resolve-Path -LiteralPath $DocumentRoot -ErrorAction Stop).Path
$python = Join-Path $PSScriptRoot "..\myenv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "The project Python environment was not found at: $python"
}

# These variables are inherited by Label Studio and intentionally apply only to
# this launcher process, rather than enabling arbitrary local-file access system-wide.
$env:LOCAL_FILES_SERVING_ENABLED = "true"
$env:LOCAL_FILES_DOCUMENT_ROOT = $resolvedRoot
# Some managed Windows environments define DEBUG as "release", which Label
# Studio does not accept because it requires a boolean value.
$env:DEBUG = "false"

Write-Host "Label Studio local-file root: $resolvedRoot"
Write-Host "Opening Label Studio at http://localhost:8080"
# Invoke the console entry point through Python because Windows Application
# Control may block the generated label-studio.exe wrapper.
& $python -c "from label_studio.server import main; main()" start
