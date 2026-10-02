param([int]$F0, [int]$F1, [string]$Label, [string]$Rec)
# One render worker (car_movie --render --frames F0 F1) of the recording $Rec under py-spy and
# the memory sampler, into runs\perf\<Label> (part.mp4, spy.txt, mem\, run.log).
. "$PSScriptRoot\env.ps1"
if (-not $Rec) { $Rec = "$data\rec8_ref\rec" }
$env:NUMBA_NUM_THREADS = "1"
$out = "$data\$Label"
New-Item -ItemType Directory $out -Force | Out-Null
$look = "--size3d 2560 1440 --width2d 1900 --ssaa 2 --fps 30 --crf 14 --preset slow --orbit 25 --cam-dist 12 --drop-scale 4"
$p = Start-Process -FilePath $py -PassThru -WindowStyle Hidden -WorkingDirectory $repo `
  -ArgumentList "-u examples\car_movie.py --render `"$Rec`" $look --frames $F0 $F1 --part `"$out\part.mp4`" --part-id 0" `
  -RedirectStandardOutput "$out\run.log" -RedirectStandardError "$out\run.err"
$child = Get-ChildPython $p
$s = Start-Process -FilePath $py -PassThru -WindowStyle Hidden -WorkingDirectory $repo `
  -ArgumentList "`"$sampler`" --pid $($p.Id) --out `"$out\mem`" --every 2 --dump-step 4 --min-c 15"
$r = Start-Process -FilePath $spy -PassThru -WindowStyle Hidden `
  -ArgumentList "record --pid $child --rate 20 --nonblocking --format raw --output `"$out\spy.txt`"" `
  -RedirectStandardOutput "$out\spy.log" -RedirectStandardError "$out\spy.err"
$t0 = Get-Date
$p.WaitForExit()
Start-Sleep -Seconds 5
"$Label done: exit $($p.ExitCode), {0:N1} min at $(Get-Date -Format HH:mm:ss)" -f ((Get-Date) - $t0).TotalMinutes
