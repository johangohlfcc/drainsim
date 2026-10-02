# Paths and settings shared by the profiling scripts; dot-source it: . "$PSScriptRoot\env.ps1"
#   $home_   this repository             $repo  the drainsim to run ($env:DS_REPO, default: this one)
#   $py, $spy  the venv's python and py-spy ($env:DS_VENV, default: the .venv next to the repository)
#   $data    runs\perf (results, captures, the 8 mm reference recording)
#   $sampler the memory sampler          $car   the car mesh (xc90.stl in the repository)
$home_ = (Resolve-Path "$PSScriptRoot\..\..").Path
$repo = if ($env:DS_REPO) { $env:DS_REPO } else { $home_ }
$venv = if ($env:DS_VENV) { $env:DS_VENV } else { (Resolve-Path "$home_\..\.venv").Path }
$py = "$venv\Scripts\python.exe"; $spy = "$venv\Scripts\py-spy.exe"
$data = "$home_\runs\perf"
$sampler = "$home_\tools\runs\mem_sampler.py"
$car = "$home_\xc90.stl"
$env:PYTHONPATH = "$repo;$repo\examples"
$env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
Remove-Item Env:OMP_NUM_THREADS -ErrorAction SilentlyContinue
$env:NUMBA_NUM_THREADS = "16"

function Get-ChildPython($p) {
  # the venv's python.exe is a launcher: the interpreter runs in its child process
  Start-Sleep -Seconds 2
  $c = (Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" | Where-Object Name -eq "python.exe" | Select-Object -First 1).ProcessId
  if ($c) { $c } else { $p.Id }
}
