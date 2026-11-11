"""Port check: rows of vdd_steer.py --steer resid alone vs the stored vdd_decode.py rows of the same configuration.
usage: port_check.py NEW.jsonl STORED.jsonl"""
import json, sys
new = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
old = {r['id']: r for r in (json.loads(l) for l in open(sys.argv[2]) if l.strip())}
same = sum(r['prediction'] == old[r['id']]['prediction'] for r in new)
print(f"port check (vdd_steer resid alone vs stored vdd_decode Full): identical {same} of {len(new)}; tokens new/stored",
      [(r['generated_tokens'], old[r['id']]['generated_tokens']) for r in new])
