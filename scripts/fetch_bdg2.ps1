param(
    [string]$Destination = "$PSScriptRoot\..\vendor\bdg2"
)

$resolvedParent = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\vendor"))
New-Item -ItemType Directory -Force -Path $resolvedParent | Out-Null

if (-not (Test-Path (Join-Path $Destination ".git"))) {
    git clone --depth 1 --filter=blob:none --no-checkout `
        https://github.com/buds-lab/building-data-genome-project-2.git `
        $Destination
}

git -C $Destination sparse-checkout init --no-cone
git -C $Destination sparse-checkout set `
    README.md `
    data/metadata/metadata.csv `
    data/meters/cleaned/electricity_cleaned.csv
git -C $Destination checkout master

Write-Host "BDG2 files are available at $Destination"

