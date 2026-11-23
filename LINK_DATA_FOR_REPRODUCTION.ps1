param([Parameter(Mandatory=$true)][string]$DatasetRoot)
$resolvedDataset = (Resolve-Path -LiteralPath $DatasetRoot -ErrorAction Stop).Path
if (!(Test-Path -LiteralPath (Join-Path $resolvedDataset 'train.csv')) -or !(Test-Path -LiteralPath (Join-Path $resolvedDataset 'train_images'))) { throw 'Expected authorized Severstal train.csv and train_images' }
$linkRoot = Join-Path $PSScriptRoot 'data'
$linkPath = Join-Path $linkRoot 'severstal-steel-defect-detection'
if (Test-Path -LiteralPath $linkPath) { throw 'Data location exists; preserve it' }
New-Item -ItemType Directory -Path $linkRoot -Force | Out-Null
New-Item -ItemType Junction -Path $linkPath -Target $resolvedDataset | Out-Null
Write-Output 'Existing dataset linked; images were not copied. Do not delete or alter source data.'
