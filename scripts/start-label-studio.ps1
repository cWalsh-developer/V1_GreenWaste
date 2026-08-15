[CmdletBinding()]
param(
    [string]$DocumentRoot = (Join-Path $PSScriptRoot "..\data\raw")
)

$resolvedRoot = (Resolve-Path -LiteralPath $DocumentRoot -ErrorAction Stop).Path
$labelStudio = Join-Path $PSScriptRoot "..\myenv\Scripts\label-studio.exe"

if (-not (Test-Path -LiteralPath $labelStudio -PathType Leaf)) {
    throw "Label Studio was not found at: $labelStudio"
}

# These variables are inherited by Label Studio and intentionally apply only to
# this launcher process, rather than enabling arbitrary local-file access system-wide.
$env:LOCAL_FILES_SERVING_ENABLED = "true"
$env:LOCAL_FILES_DOCUMENT_ROOT = $resolvedRoot

Write-Host "Label Studio local-file root: $resolvedRoot"
Write-Host "Opening Label Studio at http://localhost:8080"
& $labelStudio start
