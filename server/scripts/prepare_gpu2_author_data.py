"""Cache author caption pairs and disjoint, annotation-derived calibration."""
import hashlib,json,urllib.request,concurrent.futures
from pathlib import Path
ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def main():
    pointer=json.loads((ROOT/'outputs/gpu2-ephs-chair-current.json').read_text());output=Path(pointer['output']);data=output/'data';images=data/'author-images';images.mkdir(exist_ok=True)
    source=ROOT/'upstream-author-20261027-revis-vti/VTI/experiments/data/hallucination_vti_demos.jsonl'
    tests=read(ROOT/'data/hallucination-suite/pope.jsonl')+read(ROOT/'data/hallucination-suite/chair.jsonl');testids={int(r['group_id'].split(':')[-1]) for r in tests};testhashes={r['image_sha256'] for r in tests}
    demos=[x for x in read(source) if int(x['id']) not in testids][:32];assert len(demos)==32
    def download(row):
        path=images/row['image']
        if not path.exists():
            url='https://s3.amazonaws.com/images.cocodataset.org/train2014/'+row['image'];tmp=path.with_suffix('.partial');urllib.request.urlretrieve(url,tmp);tmp.replace(path)
        from PIL import Image
        with Image.open(path) as img:img.verify()
        assert sha(path) not in testhashes
        return {**row,'image_path':str(path),'image_sha256':sha(path)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:demos=list(pool.map(download,demos))
    (data/'author-preferences.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in demos))
    subset=json.loads((data/'instances_dev.json').read_text());categories={r['id']:r['name'] for r in subset['categories']};by={}
    for a in subset['annotations']:by.setdefault(a['image_id'],set()).add(a['category_id'])
    frequency={c:sum(c in cats for cats in by.values()) for c in categories}
    fitted={r['group_id']:r for r in read(data/'fit.jsonl')};cal=[]
    for group,row in sorted(fitted.items()):
        imageid=int(group.split(':')[-1]);present=by[imageid];positive=sorted(present,key=lambda c:(-frequency[c],c))[:3];negative=sorted(set(categories)-present,key=lambda c:(-frequency[c],c))[:len(positive)]
        for label,classes in [('yes',positive),('no',negative)]:
            for category in classes:
                cal.append({'id':f'author-calibration:{imageid}:{category}','group_id':group,'image':row['image'],'image_sha256':sha(row['image']),'label':label,'prompt':f'Is there a {categories[category]} in the image? Please answer yes or no.','category_id':category,'split':'tune'})
    assert {r['image_sha256'] for r in cal}.isdisjoint(r['image_sha256'] for r in demos)
    assert {r['image_sha256'] for r in cal+demos}.isdisjoint(testhashes)
    (data/'author-calibration.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cal))
    provenance={'implementation':'author-operations-gemma-bridge','author_reproduction':False,'preference_source':str(source),'preference_source_sha256':sha(source),'preference_images':len(demos),'calibration_images':len(fitted),'calibration_questions':len(cal),'development_test_id_overlap':0,'development_test_hash_overlap':0,'vti_mask_trials':50,'vti_mask_ratio':0.99,'vti_alpha_text':0.9,'vti_alpha_vision':0.9,'revis_alpha':1.6,'revis_gamma':1.0,'revis_factual_percentile':90.0,'revis_layer_rule':'deepest layer >=10 with positive FP-minus-TP/TN risk separation','notes':['Author caption pairs reused for both REVIS and VTI.','Gemma processor, model module paths and dtype restoration are explicit architecture bridges.','32 author demonstrations used as a fixed extraction budget; the VTI example CLI uses 70.']}
    (output/'author-development.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(provenance,indent=2))
if __name__=='__main__':main()
