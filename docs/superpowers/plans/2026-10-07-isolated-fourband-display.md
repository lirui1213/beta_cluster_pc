# Isolated Four-Band Display Implementation Plan

> **For agentic workers:** Execute the approved design task-by-task with test-driven development and a final independent code review.

**Goal:** Keep Windows native brain/curve rendering outside the Jupyter classification kernel without changing labels or controls.

**Architecture:** The Notebook serializes a trusted local snapshot of its current extraction, fitted model, source paths and display settings. A fresh Python subprocess owns all native brain/curve windows; a separate numerical process owns filtering, smoothing and the session cache. Windows Job ownership and a launch gate cover the entire process tree before any descendants start. The parent waits without creating brain/curve GUI objects and reports abnormal exits with persistent logs.

**Tech Stack:** Existing Python environment, subprocess, joblib, Tk, Matplotlib, PyVista.

## Global Constraints

- Preserve user Notebook parameters, existing saved outputs, manual K, exclusions and classification algorithms.
- Preserve both ordinary and selectable brain views, channel label toggle, right-click channel/curve linking, display colors/opacity and explorer controls.
- Do not install packages, modify the environment, or push GitHub.
- Keep worker snapshots and logs under each round's `display_sessions` directory; load only trusted local joblib files.
- Native GUI objects and the cache must not be serialized from the kernel.

### Task 1: Process Boundary

Files: `3-bands/fourband/display_process.py`, `display_worker.py`, `display_computation.py`, `display_lifecycle.py`, `display_logging.py`, `tests/test_fourband_display_process.py`, `tests/test_fourband_display_computation.py`.

- [x] Write tests for local snapshot round-trip, same interpreter, non-shell launch, child error isolation, interrupt cleanup, missing paths and disabled displays.
- [x] Run tests and observe the missing API failure.
- [x] Implement `run_display(extraction, result, *, output_dir, fif_path=None, onset_path=None, offset_path=None, wav_path=None, surface_options=None, show_surface=False, show_selectable=False, link_explorer=True, explorer_settings=None, cache_max_mb=512, explorer_only=False)` returning session directory/PID/return code.
- [x] Worker loads its own recording, events, audio and cache only when curves are requested; routes callbacks and label toggles to existing rendering functions; closes explorer in finally.
- [x] Verify failure reports contain logs and do not invalidate the parent's model.

### Task 2: Notebook Integration

Files: `3-bands/ECoG_4band_clustering_pipeline.ipynb`, `3-bands/create_notebook.py`, relevant notebook tests, `3-bands/README.md`.

- [x] Add tests that execute the brain/explorer cells with a captured launcher, asserting guards, parameter forwarding and no in-kernel window creation.
- [x] Replace native brain and linked explorer creation with `run_display`; use the same boundary for standalone explorer.
- [x] Update generation/migration paths so regeneration cannot restore the unsafe display workflow.
- [x] Preserve user settings and unrelated cells/outputs; document blocking behavior, startup cost, retained logs and interruption.

### Task 3: Verification And Local Finish

- [x] Run focused tests and a fresh Python subprocess with the actual HS0015/B03 saved extraction/model and current anatomy/settings.
- [x] Exercise native right-click linking, curve completion, screenshots and closing both brain views from a diagnostic-only wrapper, not test branches in production.
- [x] Check parent survival after an intentional child process crash.
- [x] Attempt actual Jupyter kernel verification and explicitly report any ZeroMQ limitation.
- [x] Run full suite serially, inspect artifacts, obtain code review, check Notebook schema/cell compilation and `git diff --check`.
- [x] Commit only scoped changes locally; no push.

## Verification Results

- Full suite: 548 passed with `OMP_NUM_THREADS=1`, `--capture=sys`.
- Final pre-commit full-suite rerun: 548 passed in 95.00 seconds. An intervening rerun had one legacy same-process Tk render timeout with cross-thread destructor warnings (547 passed); the exact GUI test passed separately, then the complete suite passed again. This intermittent legacy path remains a residual risk outside the isolated Notebook display route.
- Actual ECoG environment: re-executed original Notebook cells 1-10 on HS0015/B03 with saved 170 manual trial exclusions and K=10 then K=4. Both saved assignment tables are identical to the original run.
- Full workflow log: `3-bands/outputs/verification/kernel_reproduction/20261007_030952_480327/execution.log`.
- Final consoleless numerical-process brain probe: `3-bands/outputs/verification/isolated_display/20261007_031844_510747/`; real VTK picks, two native surfaces, separate GUI/numerical PIDs, clean logs.
- Final standalone explorer probe: `3-bands/outputs/verification/isolated_display/20261007_032314_325260/`; two channel draws, saved plot, normal close, clean GUI/numerical error logs.
- Child crash and full descendant interruption tests pass; the descendant test also fails when Job cleanup is disabled in memory.
- Independent review findings (startup interruption, idle numerical failure, startup numerical failure, migration ownership) have regression tests and no remaining findings in scoped rereview.
- Actual Jupyter launch is still blocked by local ZeroMQ `Bad file descriptor (epoll.cpp:73)`. The successful original-cell execution is in-process, not an actual Jupyter kernel test.
