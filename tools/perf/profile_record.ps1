param([string]$Dx = "0.008", [string]$Label = "rec8")
# A full car_movie --record run (setup + all steps + the field writer) under py-spy and the
# memory sampler, into runs\perf\<Label> (rec\ the recording, spy.txt, mem\, run.log).
# Analyse: analyze_record.py, writer_lines.py, par_callers.py, film_breakdown.py <dir>\spy.txt;
# compare with the reference: compare_rec.py runs\perf\rec8_ref\rec <dir>\rec.
. "$PSScriptRoot\env.ps1"
$out = "$data\$Label"
New-Item -ItemType Directory $out -Force | Out-Null
$t0 = Get-Date
$p = Start-Process -FilePath $py -PassThru -WindowStyle Hidden -WorkingDirectory $repo `
  -ArgumentList "-u examples\car_movie.py --stl `"$car`" --dx $Dx --levels 3 --subcells 2 --narrow 4 --threads 1 --rotation pitch --sense 1 --display-faces 2000000 --record `"$out\rec`"" `
  -RedirectStandardOutput "$out\run.log" -RedirectStandardError "$out\run.err"
$child = Get-ChildPython $p
$s = Start-Process -FilePath $py -PassThru -WindowStyle Hidden -WorkingDirectory $repo `
  -ArgumentList "`"$sampler`" --pid $($p.Id) --out `"$out\mem`" --every 5 --dump-step 1000 --min-c 15"
$r = Start-Process -FilePath $spy -PassThru -WindowStyle Hidden `
  -ArgumentList "record --pid $child --rate 20 --nonblocking --threads --format raw --output `"$out\spy.txt`"" `
  -RedirectStandardOutput "$out\spy.log" -RedirectStandardError "$out\spy.err"
$p.WaitForExit()
Start-Sleep -Seconds 5
"$Label done: {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes
