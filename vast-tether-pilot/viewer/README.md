# J-lens top-100 viewer (local, no server)

Open `index.html` in Chrome or Edge (double-click). Data: Qwen3.5-4B, 100 COCO pilot images, Jacobian lens and logit lens,
31 layers, top 100 tokens at every position (vast box `/workspace/tether-pilot/out/jlens_pos100`, from
`pilot/jlens_positions_top100.py`).

- The readout at position p is the lens on the residual at p, i.e. the prediction for the token at p + 1.
  Clicking a caption token opens the step that writes it; image tokens are read at their own position.
- `data/<id>.js` is loaded only when that image is opened (about 15 MB each).
- Rebuild: copy `jl_export.py` to the box and run
  `ssh <vast box> 'cd /workspace/tether-pilot; nice -n 10 .venv35/bin/python out/jl_export.py' > full.tar`,
  then `tar xf full.tar` here. The script streams a tar to stdout and writes nothing on the box.
