L=/home/project-b/LJW; T=$L/bench/THRONE/throne; O=$L/vlm-grounding-steering/outputs/rbd2-20261102; COCO=$L/bench/coco2017/annotations/instances_val2017.json
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=; cd $T
for spec in "Qwen2.5-VL-7B throne_full1000_2judges base s16a1bl2p1 revis rudder_clone grains_clone vti_author" "Gemma-4-E4B gemma_throne1000_2judges base s24a1bl2p1 revis rudder_clone grains_clone vti"; do
  set -- $spec; B=$1; D=$2; shift 2
  for t in "$@"; do
    echo "##### OFFICIAL 2-judge | $B | $t | files: $(ls $O/diag/$D/$t/combined | sed 's/evaluations__home_project-b_LJW_bench_models_//' | tr '\n' ' ')"
    for s in micro class; do $L/venvs/throne/bin/python throne_score_aqa.py --coco_val_ann_path $COCO --model_eval_path $O/diag/$D/$t/combined --thresholds 4 5 6 --metric_strategy $s 2>&1 | grep -vE 'loading annotations|^Done|creating index|index created'; done
  done
done
for t in base revis rudder_clone grains_clone vti_author; do
  echo "##### OFFICIAL 3-judge (xl) | Qwen2.5-VL-7B | $t | files: $(ls $O/runs/throne_full1000/eval/$t/combined | sed 's/evaluations__home_project-b_LJW_bench_models_//' | tr '\n' ' ')"
  grep -vE 'loading annotations|^Done|creating index|index created' $O/runs/throne_full1000/eval/$t/official_scores.txt
done
