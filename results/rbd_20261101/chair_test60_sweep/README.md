# chair_test60 sweep (queued 2026-11-01 17:37 UTC)

Requested by the user after `main` lost recall on chair_test60. Every round-2 dev config plus two new ones run on
chair_test60 with `scripts/rbd2/sweep_decode.py`; output goes to `/workspace/vdd/chair_test60_sweep/` on the server.
The base and main rows are copied from `../chair_test60` (same decoding path), so all runs share one Base.

**This is a sweep on the test set.** A config picked from it is selected on chair_test60, so chair_test60 is not held
out for that config. A final claim needs CHAIR test images that no run has used (for example CHAIR test 100-499).

New option `--bpos`: the visual-evidence belief only raises objects (lower clip 0 instead of -clip). With `main`,
recall on chair_test60 fell from 68.4 to 63.6. bpos removes both parts that push objects down (the penalty and the
negative side of the belief) to test whether steering plus raising objects with visual evidence can lower CHAIR_s
without that recall loss. It has not been run on dev.

| queue | configs, in order |
|---|---|
| `q0.txt` (GPU 0) | s16a3, bpos_s16a3, s16a1bl1, s16a3bl1, bl1, s16a2bl1p05 |
| `q1.txt` (GPU 1) | bpos_s16a1, s16a1, pen1, s16a2bl1, s20a2bl1, s16a1bl2p1 |

`scripts/rbd2/sweep.sh GPU QUEUE` waits until the GPU has no round-2 `worker.sh`/`vrun.sh` process and less than
1 GB in use, then runs the queue; logs go to `/workspace/logs/sweep/`. `sweep_decode.py` is `vdd_decode.py` with a
four-line `--bpos` patch; without `--bpos` it decodes identically.

Status: unknown. The sweep was waiting behind the round-2 test queue (amber_test60 was running at 17:37 UTC, THRONE
next), and the server was unreachable on 2026-11-02 at 02:40 UTC, so no sweep rows or scores are in this commit.
