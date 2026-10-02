# Profiling and benchmark tools

Scripts used for the 7.0 performance work (see `REVIEW_v70.md`). Their data goes to `runs\perf\`
in the repository folder (not in git). The PowerShell scripts share `env.ps1`. `DS_REPO` points
them at another checkout to measure, e.g. a worktree of an older commit for a before/after
comparison:

    git worktree add --detach ..\drainsim_base <commit>
    $env:DS_REPO = "D:\workspace\drainsim_base"

Every change is meant to give the same numbers as before (bit for bit). `verify.ps1 <label>`
runs the tests and the 20 mm car, octree and uniform, into `runs\verify\<label>`. The trapped
air and liquid lines must equal those of the version before.

## Whole runs (py-spy at 20 Hz and the memory sampler)

| script | what |
|---|---|
| `profile_record.ps1 [-Dx 0.008] [-Label rec8]` | a full `car_movie --record` (8 mm: about 25 min) into `runs\perf\<Label>` |
| `profile_run.ps1 -Dx -Label [-Steps -T0]` | the setup (`perf_setup.py`: time per part, memory per array) and optionally some steps |
| `profile_render.ps1 -F0 -F1 -Label [-Rec]` | one render worker on frames F0..F1 of a recording |
| `run_scaling.ps1` | `examples\thread_scaling.py`: 1-32 threads, the state checked identical |
| `compare_rec.py A B` | two recordings step file by step file; `runs\perf\rec8_ref\rec` is the 8 mm reference (7.0) |

Profile analysis (py-spy raw output, numba kernels attributed to the calling line):
`analyze_record.py` (setup / time loop / field writer, by part and line), `par_callers.py` (the
`par.dual` kernels by caller, the fill-spill by line), `film_breakdown.py` and `film_lines.py`
(film update), `writer_lines.py` and `fields_lines.py` (the movie field writer),
`spy_summary.py` (any profile).

## Kernel benchmarks without the setup

Real inputs captured once from the 8 mm car (`runs\perf\captures`), replayed in seconds, with
the results checked bit for bit against the capture:

| capture | replay | what |
|---|---|---|
| `capture_fsm.py` | `bench_fsm.py dip_0 dip_1 --threads 16 1` | `fill_spill` calls of dip and drip-off steps, time per phase (`fsm.TIMING`) |
| `capture_film.py` | `bench_film.py dip12_0 dip30_0 hang60_1` | `FilmModel.update` calls (dry dip, wet dip, drip-off), time per kernel |
| - | `bench_writer.py` | the recording's field writer, old against new code, on states of a run |
| - | `fill_counts.py dip_0` | work counters of the fill cascade on a capture |

A capture holds the state that the code of its time reads. After a change to what `fill_spill`
or `FilmModel` keep, capture again.
