# No-anchor steering search (user 2026-11-10 ~22:20 KST, /loop)
User: GPUs 0 and 5 only; no anchor; keep Eq. steer (h~ = h + alpha s(t) (h - h_gray) at decoder layer L, row 0, every step t>=1,
gray row untouched); make steering effective on CHAIR test; belief only a supporting role (weaker belief hyperparameters allowed);
Qwen2.5-VL-7B and Gemma 4 E4B only; CHAIR test only, NO dev data. No Base / stored-run re-runs.
- Weak belief for the whole search: lam 0.5, pen 0.25 (paper: lam 1, pen 0.5). Reference "ns" = same belief, alpha 0 (steer none).
- Steering knobs only: layer L, alpha, s(t) schedule (decay = 1 - r(t) | const = 1). Qwen sdpa, Gemma eager (as stored rows).
- Screen: rows 0-99 of chair_test500. Steering counts as working if CHAIR_s(steer) < CHAIR_s(ns) with recall >= ns - 1.0.
  Winners go to rows 100-499 (and 0-499 reported); significance only from rows 100-499 paired bootstrap vs ns at the same belief.
- References kept from stored 500-row runs: base, rbd_full, abl_nosteer, abl_steer (rbd2-20261102/runs/{chair_abl500_all,gabl_chair500_all}).
- User 22:3x: if layer/alpha/schedule cannot make steering work, changing the steering formula is allowed, but only as the last resort (still no anchor). User away; loop continues.
- 23:58 vdd_steer.py: added --ssched early (s = r(t)); decay/const code paths unchanged (backup vdd_steer.py.bak-before-early).
- 00:27 confirmation started: qwen L12a6 and gemma L18a2 vs ns on rows 100-499 (100-row chunks, placed first in the queue); round 2 neighbours appended.
- 01:31 qwen L12a6 on 164 confirmation rows: CHAIR_s 28.0 vs ns 28.0, CHAIR_i 6.95 vs 8.59, recall -0.7, 27 tokens shorter (screen CHAIR_s gain did not carry over so far). Queue reordered: gemma confirmation first, qwen round-2 screen before qwen L12a6 rows 300-499.
- 02:03 interim: qwen L12a6 rows 100-299 CHAIR_s 28.0 vs ns 27.5 (CHAIR_i 7.37 vs 8.41, recall -0.9, 26 tok shorter); gemma L18a2 104 rows 32.7 vs 33.7, CHAIR_i lower, recall +1.9, 53 tok longer. Screen gains shrink (selection effect).
- 02:45 LAST RESORT (user-approved formula change, no anchor): --steer restore. Eq. steer with alpha replaced by
  alpha_t = clip(kappa * m0 / |d_t| - 1, 0, cap), m0 = mean |d_u| over caption steps u <= 16 (no steering while measured),
  d_t = h - h_gray at layer L as before; s(t) decay unchanged. Reading: the image contribution is brought back to kappa x its
  opening norm along its current direction ("restore the image's contribution once its direct influence has faded").
  Screen on rows 0-199 vs ns (same weak belief). Candidates: qwen L16 kappa 1 / 1.5 (cap 6), gemma L24 / L18 kappa 1 (cap 6).
  Knob-only results so far are kept and reported (held-out: no CHAIR_s change; CHAIR_i lower; gemma recall +2).
- 03:31 qwen restore L16 k1 (rows 0-199): CHAIR_s 26.5 vs ns 28.0 [-6,+3], recall +1.5 [-0.3,+3.5], same length; k1.5 (132 rows) recall +3.4 [+0.5,+6.6], CHAIR_s 25.8 vs 26.5. Pre-set: R16k15 goes to rows 200-499 (qwen ns 300-499 re-added); primary there = CHAIR_s and recall vs ns. gemma L18a2 held-out 200 rows: 36.5 vs 37.5, recall +2.0.
- 04:05 R16k15 full rows 0-199: CHAIR_s 31.5 vs 28.0 (+3.5), recall +2.3; rows 200-299 (91) 36.3 vs 28.6 -> stronger kappa trades hallucination for recall; R16k15 300-499 and R16k2 dropped. R16k1 (0-199: 26.5 vs 28.0, recall +1.5) goes to rows 200-499 instead (deviation from the 03:31 pre-set, decided on the full 200-row screen).
- 04:39 Interim 0-499: qwen R16k1 (392 rows) CHAIR_s 28.8 vs ns 28.8, recall +1.5 [0.0,+3.1]; gemma L18a2 (300) 33.3 vs 36.7 [-9,+2], recall +1.7. Added: the new steering with the paper's belief (lam 1, pen 0.5): qwen R16k1b1 rows 0-499, gemma L18a2b1 rows 0-199, compared with stored lam-1 abl_nosteer / rbd_full (different decode run, same GPU type not guaranteed).
- 05:13 qwen R16k1 (lam .5) final 500: CHAIR_s 29.4 vs ns 29.2 [-3.0,+3.4], recall +1.0 [-0.4,+2.3] -> null. R16k1b1 (lam 1) 225 rows: 25.8 vs stored abl_nosteer 26.2, recall +2.1 [+0.4,+3.9]; vs stored rbd_full 23.1 (CHAIR_i sig worse) -> restore is worse than the paper's resid L16 a2 at lam 1; its rows 300-499 dropped. Added lam-1 resid screens qwen L12a6b1, L16a3b1 (rows 0-199).
- 05:45 gemma L18a2 (lam .5) 402 rows: CHAIR_s 33.1 vs ns 34.8 [-6.5,+3.2], recall +1.8 [+0.1,+3.6] sig, 52 tok longer. qwen R16k1b1 400 rows: 27.5 vs stored abl_nosteer 28.0, recall +1.4 [-0.1,+2.8]; vs rbd_full 25.5 (worse). Queue reordered (see queue.txt header); restore screens dropped.
- 10:12 qwen L16a3b1 (lam 1, resid L16 alpha 3) rows 0-199: CHAIR_s 23.5 vs stored rbd_full 26.5 [-8.0,+2.0] and abl_nosteer 28.0 [-9.5,+1.0]; recall 68.4 vs 68.8. Pre-set: rows 200-499 decide (vs rbd_full on the same rows). gemma L18a2b1 500 rows 29.8 vs rbd_full 26.2 (worse).
- 10:17 user "5만써야될듯": GPU 0 workers stopped; only w5a/w5b on GPU 5 from now.
## 2026-11-11 10:21 new loop (user): GPU 5 only, Gemma E4B only, CHAIR test only, no dev, no anchor; small formula changes allowed
if needed; belief may stay weak; no comparison with old scores needed; success = Full better than the stored Base AND a clear steering
effect (Full vs ns at the same belief). Qwen L16a3b1 rows 300-499 stopped (rows 0-199: 23.5 vs stored rbd_full 26.5, but stored rows come
from vdd_decode.py on another decode run; 12/200 captions already differ in the first 5 words where steering and belief are 0, so the
gap mixes decode noise with the alpha effect - unresolved).
Gemma plan (weak belief lam .5 pen .25, ns rows 0-499 exist): 200-row screens. L24a2 (rows 0-99 30.0 vs 35.0) completed to 200;
blur contrast L18a2bl (h_bar = Gaussian-blurred copy, r=12, belief still uses gray: keeps the scene gist in the contrast so d_t carries
object detail - hallucinations are mostly scene-consistent co-occurrences); stronger L18a3.
- 10:55 gemma L24a2 (lam .5 pen .25, resid L24 alpha 2) rows 0-147: CHAIR_s 28.4 vs ns 35.8 [-13.5,-1.4], CHAIR_i 6.47 vs 7.02, recall -1.6 [-4.8,+1.5], same length; vs stored Base 34.5. Pre-set before rows 148-499: confirmation = rows 200-499 vs ns (primary CHAIR_s, paired bootstrap 95% CI; recall reported). Then dose L24a3 (rows 0-199).
- 11:27 gemma L24a2 new rows 100-191 (not used for its selection): 27.2 vs ns 37.0 [-19.6,-1.1], CHAIR_i 5.42 vs 5.39, recall -3.2 [-7.0,+0.4] (fewer objects named at equal length). Blur contrast L18a2bl rows 0-69: 27.1 vs 35.7 [-20,+2.9], recall +3.2, CHAIR_i 8.31 vs 9.52, longer. Queue interleaved.
- 14:39 gemma L24a2 final 500: CHAIR_s 31.8 vs ns 34.0 [-6.0,+1.4], CHAIR_i 6.21 vs 6.89 [-1.59,+0.18], recall 63.3 vs 63.2; held-out rows
  200-499: 33.0 vs 32.0 [-4.0,+5.7] -> NOT confirmed (effect only on the selection rows 0-199). vs stored Base 34.6: -2.8 [-6.6,+1.2].
  Gemma L18a2 (gray) 500: 32.6 vs 34.0, recall +2.0 [+0.3,+3.5]; L18a2bl 200: 33.0 vs 37.0 [-11,+3]. L24a3 66 rows: CHAIR_i up.
  Power note: paired CHAIR_s SE at n=500 is ~2.4 points, so only steering effects of >= ~5 points can reach significance.
- 15:13 vdd_steer.py: --ssched gate (backup vdd_steer.py.bak-before-gate). Per decode step the logits processor records
  KL(p_real || p_gray) over the full vocabulary (raw logits, before belief); steering at the next step uses s = 1 - r(t) only if that
  KL is below the caption's running mean, else 0 ("steer where the image is read less", per caption instead of the average r(t)).
  Screen gemma L24 alpha 3 gate on rows 0-199. Other paths unchanged.
- 15:35 user: weak belief deliberate; asks whether the steering effect holds with the paper belief -> gemma L24a2b1 (lam 1 pen 0.5, L24 a2) rows 0-499 queued before the gate screen; compared with stored abl_nosteer / rbd_full / base (other decode run: noisier comparison).
- 15:38 user plan: stronger-but-not-full belief + this steering, then the ablation table. Step 1 (rows 0-199): L24a2m / nsm at lam .75
  pen .35. If the steering gap there is >= the lam .5 gap, extend to 500 + w/o belief / w/o penalty at that setting; otherwise the
  ablation is completed at lam .5 pen .25 (abl_nobelief_L24a2, abl_nopen_L24a2 queued; Full = L24a2, w/o steering = ns already 500).
  Hyperparameters are chosen on test rows (no dev by user rule) -> must be disclosed. join_gpu0.sh adds w0a/w0b on GPU 0 when no other
  user's process is on it (GPU 0 now = HTR jobs).
- 15:45 user: the belief-tuned ablation plan runs on a new vast.ai box instead; GPU 5 keeps the exploration loop. Those units commented out here; join_gpu0.sh stopped.
- 15:43 L24a3 192 rows: 31.8 vs ns 35.9 [-10.9,+2.6], CHAIR_i up (7.03 vs 6.68) -> no better than L24a2. L18a3 46 rows: 30.4 vs 28.3, captions 447 tok (+88) -> dropped its rows 100-199. Next: gate screen, then 2-layer.
- 17:50 qwen L16a3b1 on the 3080 vast box, all 500: 24.0/6.58/67.0/171 vs stored -steering 27.6/7.79/67.4/191, TETHER(a2) 25.6/6.86/68.9/178, Base 30.4/8.40/69.8/189. Chain on box: abl_nopen_a3 then abl_nobelief_a3 (500 each). Gemma gate L24a3g rows 0-199 scored (see chat).
- 17:53 gate (L24a3g) rows 0-199: CHAIR_s 37.5 vs ns 37.0 [-6,+7] -> no effect. Queued: L24a2bl (blur contrast at L24), sel24r02 (belief-guided direction, rho .2) on rows 0-199.
- 20:25 vast lam .75 step 1 stopped at 99/100 rows (Full L24a2m 35.4 vs w/o steering nsm 29.3, dS [-1.0,+13.1]: steering hurts under stronger belief). User: run the belief/penalty ablation of the GPU-5 setting (lam .5 pen .25 L24 a2) on vast: abl_nopen_L24a2 (pen 0), abl_nobelief_L24a2 (lam 0, pen .25), rows 0-499 interleaved, 2080 Ti ~60 s/row -> ~17 h. Hardware differs from the GPU-5 Full / -steering rows (A6000).
- 20:25 user: Gemma loop stopped (GPU 5 pool killed, STOP set). GPU 5 now runs Qwen abl_nobelief_a3 rows 300-499 (A6000, 2 workers) while the 3080 box does rows 0-299.
- 20:32 GPU 5 Gemma exploration loop ended by the user (other session stopped w5a/w5b at 20:24 for the Qwen ablation). Last partial screens vs ns: L24a2bl 197 rows 33.5 vs 36.5 [-9.1,+3.6], CHAIR_i 6.18 vs 7.12; sel24r02 63 rows 30.2 vs 34.9, CHAIR_i 7.80 vs 10.04 [-4.78,-0.03]. Gemma ablation (-penalty/-belief) continues on vast.
