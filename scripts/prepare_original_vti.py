"""Fetch only the original VTI demonstration images and seal the new subrun."""
import concurrent.futures,hashlib,json,random,time,urllib.request
from pathlib import Path
from PIL import Image
from run_qwen_original_vti import ROOT,PARENT,AUTHOR,COMMIT,sha,save,read,verify_author

def main():
    out=ROOT/'outputs'/('qwen-original-vti-'+str(int(time.time())))
    out.mkdir();(out/'data/train2014').mkdir(parents=True)
    demos=random.Random(42).sample(read(AUTHOR/'experiments/data/hallucination_vti_demos.jsonl'),70);assert len(demos)==70
    eval_ids={r['metadata']['image_id'] for r in read(PARENT/'data/chair.jsonl')}
    eval_ids|={int(Path(r['image']).stem.rsplit('_',1)[-1]) for r in read(PARENT/'data/pope.jsonl')}
    assert not eval_ids&{int(d['id']) for d in demos}
    def fetch(d):
        image=out/'data/train2014'/d['image'];url='https://s3.amazonaws.com/images.cocodataset.org/train2014/'+d['image']
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=35) as source:image.write_bytes(source.read())
                with Image.open(image) as im:im.verify()
                return {'id':d['id'],'image':d['image'],'url':url,'sha256':sha(image),'bytes':image.stat().st_size}
            except Exception:
                if attempt==2:raise
                time.sleep(1+attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:images=list(pool.map(fetch,demos))
    save(out/'demo-manifest.json',{'author_demos_sha256':sha(AUTHOR/'experiments/data/hallucination_vti_demos.jsonl'),
        'selection':'original author random.sample of 70 demos with seed 42','images':images,'eval_image_id_overlap':0})
    parent=json.loads((PARENT/'protocol.json').read_text())
    protocol={'status':'sealed_original_author_vti_subrun','parent_output':str(PARENT),'parent_protocol_sha256':parent['sha256'],
        'author_commit':COMMIT,'author_sources':verify_author(),'worker_source_sha256':sha(ROOT/'scripts/run_qwen_original_vti.py'),
        'model':{**parent['model'],'dtype':'float16'},'gpu_indices':[0],'method':'vti_author',
        'data_manifest_hashes':{b:sha(PARENT/'data'/(b+'.jsonl')) for b in ['pope','chair']},
        'demo_manifest_sha256':sha(out/'demo-manifest.json'),'alpha_image':.9,'alpha_text':.9,'rank':1,
        'num_demos':70,'num_trials':50,'mask_ratio':.99,'hyperparameters_source':'author README MMHal invocation; Qwen2.5 POPE/CHAIR were not supplied by authors',
        'adaptations':['Qwen2.5 processor/native chat inputs','Qwen ViT patch grid and hidden-state output facade',
            'Qwen ViT MLP batch-axis adapter around unchanged original VTILayer','explicit separate text/vision ModuleList views',
            'FP16 weights and original half output; FP32 eager attention score arithmetic as in documented Qwen amendment'],
        'attention_amendment_sha256':sha(PARENT/'fp16-attention-amendment.json'),'created_at':time.time()}
    protocol['sha256']=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    save(out/'protocol.json',protocol)
    import shutil
    shutil.copy2(ROOT/'scripts/run_qwen_original_vti.py',out/'worker-source.py')
    save(PARENT/'original-vti-current.json',{'output':str(out),'protocol_sha256':protocol['sha256'],'created_at':time.time()})
    print(json.dumps({'output':str(out),'images':len(images),'download_bytes':sum(i['bytes'] for i in images),'protocol_sha256':protocol['sha256']}))

if __name__=='__main__':main()
