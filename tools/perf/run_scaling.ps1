param([string]$Dx = "0.008", [string]$Label = "scaling_8mm")
# Thread scaling of the car model (examples\thread_scaling.py: 1-32 threads, dip and hang steps,
# the state checked identical) into runs\perf\<Label>.json and .log.
. "$PSScriptRoot\env.ps1"
$env:NUMBA_NUM_THREADS = "32"
New-Item -ItemType Directory $data -Force | Out-Null
Set-Location $repo
& $py -u examples\thread_scaling.py --stl "$car" --dx $Dx --subcells 2 --levels 3 --narrow 4 --threads 1 2 4 8 16 32 --steps 6 --hang-steps 6 --out "$data\$Label.json" *> "$data\$Label.log"
"exit $LASTEXITCODE"
