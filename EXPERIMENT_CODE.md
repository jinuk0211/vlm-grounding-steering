# Experiment code snapshot

This branch preserves the code used for the Gemma 4 E4B and Qwen2.5-VL experiments. The scripts retain their original server-relative paths and run-specific assumptions. They are an experiment snapshot, not a self-contained portable reproduction.

- RUDDER and GrAInS call pinned author code through `scripts/author_clone_bridge.py`, with model adapters. They require the original Git checkouts at `upstream-author-20261026/RUDDER` and `upstream-author-20261026/GrAInS`; the author repositories are not vendored here.
- Qwen original-author VTI uses `scripts/run_qwen_original_vti.py` and the original VTI checkout. Its input and tensor shapes are adapted to Qwen.
- The recorded Gemma five-profile run uses the retained local REVIS and VTI adaptations, not the original VTI implementation. The original-author VTI replacement in the recorded run applies to Qwen. REVIS is also a local adaptation in these five-profile runs.
- The Qwen parent experiment uses BF16 for Base, REVIS and RUDDER; GrAInS and original-author VTI use FP16 with FP32 attention scores. The Gemma run uses FP16 for GrAInS and BF16 for the other profiles.

Pinned upstream checkouts:
- [RUDDER](https://github.com/Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation-) `bd149d8dbde6a9097dd02aea80f83d0bd8876761`
- [GrAInS](https://github.com/duykhuongnguyen/GrAInS) `8ad78db3040fbb7b0aec9e544a3dcc7b9c050b5b`
- [VTI](https://github.com/shengliu66/VTI) `bb5dfee33a9e555f22af9faa7b3495c0ac3df69e`
- [REVIS](https://github.com/antgroup/Revis) `75f4d29d402d4c1c4dbb2c69e88e3aa50cd0c9cc`

Model weights, datasets, run artifacts, server-specific environments and some scoring and orchestration utilities are not included. Several scripts assert exact source hashes, GPU UUIDs and existing run protocols; they cannot be run unchanged on another machine. The presence of code here does not imply that the ongoing evaluation completed or that its results have been verified.
