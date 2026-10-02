param([string]$Label, [string]$Repo)
# The checks run after every change: the test suite and the 20 mm car (octree and uniform grid),
# with the drainsim in $Repo (default: this repository), into runs\verify\<Label>\ (summary.txt).
# The trapped air and liquid lines must equal those of the version before (same numbers).
if ($Repo) { $env:DS_REPO = $Repo }
. "$PSScriptRoot\env.ps1"
$out = "$home_\runs\verify\$Label"
New-Item -ItemType Directory $out -Force | Out-Null
Set-Location $repo
$t0 = Get-Date
& $py -m pytest -q tests -p no:cacheprovider *> "$out\tests.log"
"tests: exit $LASTEXITCODE, {0:N1} min" -f ((Get-Date)-$t0).TotalMinutes | Tee-Object "$out\summary.txt"
Get-Content "$out\tests.log" -Tail 3 | Tee-Object "$out\summary.txt" -Append
$t1 = Get-Date
& $py -u examples\car_article.py --stl $car --dx 0.02 --levels 2 --hang 5 --threads 1 --out "$out\check20_o" *> "$out\check20_o.log"
"octree 20 mm: exit $LASTEXITCODE, {0:N1} min" -f ((Get-Date)-$t1).TotalMinutes | Tee-Object "$out\summary.txt" -Append
$t2 = Get-Date
& $py -u examples\car_article.py --stl $car --dx 0.02 --hang 5 --threads 1 --out "$out\check20_u" *> "$out\check20_u.log"
"uniform 20 mm: exit $LASTEXITCODE, {0:N1} min" -f ((Get-Date)-$t2).TotalMinutes | Tee-Object "$out\summary.txt" -Append
Get-Content "$out\check20_o.log","$out\check20_u.log" | Select-String "setup|trapped" | ForEach-Object { $_.Line -replace "^\[car_20mm.*?\] ", "" } | Tee-Object "$out\summary.txt" -Append
