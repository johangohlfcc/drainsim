# IPS XC90 dip (RoDip-style 360 deg roll, IPS motion and bath) + 180 s drip-off, recorded into
# runs\ips_xc90\ips<dx>mm (not started automatically).   Usage: start.ps1 [dx in m, default 0.005]
# Inputs in the repository folder: xc90_10mm.stl (the car at the motion's first pose, plant
# frame), XC90_motion.xmo, bath.stl. Checks without a run: check_motion.py, final_pose.py.
param([string]$Dx = "0.005")
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$venv = if ($env:DS_VENV) { $env:DS_VENV } else { (Resolve-Path "$repo\..\.venv").Path }
$py = "$venv\Scripts\python.exe"
$run = "$repo\runs\ips_xc90"; $mm = [int]([double]$Dx * 1000); $name = "ips$($mm)mm"
New-Item -ItemType Directory $run -Force | Out-Null
$levels = if ([double]$Dx -le 0.003) { 4 } else { 3 }
$env:PYTHONPATH = "$repo;$repo\examples"
$env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:NUMBA_NUM_THREADS = "16"
Remove-Item Env:OMP_NUM_THREADS -ErrorAction SilentlyContinue
$p = Start-Process -FilePath $py -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
  -ArgumentList "-X faulthandler -u examples\car_movie.py --stl `"$repo\xc90_10mm.stl`" --motion-file `"$repo\XC90_motion.xmo`" --bath-stl `"$repo\bath.stl`" --hang 180 --dx $Dx --levels $levels --subcells 2 --narrow 4 --threads 1 --display-faces 2000000 --record `"$run\$name`"" `
  -RedirectStandardOutput "$run\$name.log" -RedirectStandardError "$run\$name.err"
"started $name recording: PID $($p.Id) at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
# the movie afterwards (NUMBA_NUM_THREADS=1; keep --stagger below the time of a part):
#   python tools\runs\render_pool.py --rec runs\ips_xc90\<name> --out runs\ips_xc90\car_dip_<name>.mp4
#       --workers 3 --chunks 8 --stagger 60 --min-free 3 --size3d 2560 1440 --width2d 1900 --ssaa 2
#       --fps 30 --crf 14 --preset slow --orbit 25 --cam-dist 12 --drop-scale 4
