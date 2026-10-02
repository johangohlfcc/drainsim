param([string]$Dx, [string]$Label, [int]$Steps = 0, [int]$Levels = 3, [int]$DumpStep = 2, [string]$T0 = "12", [string]$DtSteps = "0.1")
# Setup (and optionally $Steps model steps from t = $T0) of the car (perf_setup.py) under py-spy
# and the memory sampler, into runs\perf\<Label> (perf.json, spy.txt, mem\, run.log).
. "$PSScriptRoot\env.ps1"
$out = "$data\$Label"
New-Item -ItemType Directory $out -Force | Out-Null
$p = Start-Process -FilePath $py -PassThru -WindowStyle Hidden -WorkingDirectory $PSScriptRoot `
  -ArgumentList "-u perf_setup.py --dx $Dx --levels $Levels --steps $Steps --t0 $T0 --dt-steps $DtSteps --out `"$out\perf.json`"" `
  -RedirectStandardOutput "$out\run.log" -RedirectStandardError "$out\run.err"
$child = Get-ChildPython $p
$s = Start-Process -FilePath $py -PassThru -WindowStyle Hidden -WorkingDirectory $repo `
  -ArgumentList "`"$sampler`" --pid $($p.Id) --out `"$out\mem`" --every 2 --dump-step $DumpStep --min-c 15"
$r = Start-Process -FilePath $spy -PassThru -WindowStyle Hidden `
  -ArgumentList "record --pid $child --rate 20 --nonblocking --format raw --output `"$out\spy.txt`"" `
  -RedirectStandardOutput "$out\spy.log" -RedirectStandardError "$out\spy.err"
$p.WaitForExit()
Start-Sleep -Seconds 5
"$Label done: exit $($p.ExitCode) at $(Get-Date -Format HH:mm:ss)"
