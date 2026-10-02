# IPS XC90 dip (RoDip-style 360 deg roll, IPS motion and bath) + 180 s drip-off, recorded into
# <data>\runs\ips_xc90\<name> with a memory sampler (not started automatically).
#   start.ps1 [-Dx 0.005] [-Name ips5mm] [-Data <folder>] [-Extra "<more car_movie options>"]
# Inputs in the data folder (default: this repository): xc90_10mm.stl (the car at the motion's
# first pose, plant frame), XC90_motion.xmo, bath.stl. The code is this repository's (a worktree
# can run on the main folder's inputs with -Data). 7.1 openings, e.g.:
#   -Extra "--holes <data>\runs\openings\xc90_10mm_holes.csv --channels"
# Checks without a run: check_motion.py, final_pose.py.
param([string]$Dx = "0.005", [string]$Name = "", [string]$Data = "", [string]$Extra = "")
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
if (-not $Data) { $Data = $repo }
$venv = if ($env:DS_VENV) { $env:DS_VENV } else {
  $c = Join-Path $Data "..\.venv"; if (Test-Path $c) { (Resolve-Path $c).Path } else { (Resolve-Path "$repo\..\.venv").Path } }
$py = "$venv\Scripts\python.exe"
$run = "$Data\runs\ips_xc90"; $mm = [int]([double]$Dx * 1000)
if (-not $Name) { $Name = "ips$($mm)mm" }
New-Item -ItemType Directory $run -Force | Out-Null
$levels = if ([double]$Dx -le 0.003) { 4 } else { 3 }
$env:PYTHONPATH = "$repo;$repo\examples"
$env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:NUMBA_NUM_THREADS = "16"
Remove-Item Env:OMP_NUM_THREADS -ErrorAction SilentlyContinue
$p = Start-Process -FilePath $py -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
  -ArgumentList "-X faulthandler -u examples\car_movie.py --stl `"$Data\xc90_10mm.stl`" --motion-file `"$Data\XC90_motion.xmo`" --bath-stl `"$Data\bath.stl`" --hang 180 --dx $Dx --levels $levels --subcells 2 --narrow 4 --threads 1 --display-faces 2000000 $Extra --record `"$run\$Name`"" `
  -RedirectStandardOutput "$run\$Name.log" -RedirectStandardError "$run\$Name.err"
$p.Id | Set-Content "$run\$Name.pid"
$s = Start-Process -FilePath $py -WorkingDirectory $run -WindowStyle Hidden -PassThru `
  -ArgumentList "`"$repo\tools\runs\mem_sampler.py`" --pid $($p.Id) --out $($Name)_mem --every 15 --dump-step 10 --min-c 15" `
  -RedirectStandardOutput "$run\$($Name)_mem.log" -RedirectStandardError "$run\$($Name)_mem.err"
"started $Name recording: PID $($p.Id), memory sampler $($s.Id) at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
# the movie afterwards (NUMBA_NUM_THREADS=1; keep --stagger below the time of a part):
#   python tools\runs\render_pool.py --rec runs\ips_xc90\<name> --out runs\ips_xc90\car_dip_<name>.mp4
#       --workers 3 --chunks 8 --stagger 60 --min-free 3 --size3d 2560 1440 --width2d 1900 --ssaa 2
#       --fps 30 --crf 14 --preset slow --orbit 25 --cam-dist 12 --drop-scale 4
