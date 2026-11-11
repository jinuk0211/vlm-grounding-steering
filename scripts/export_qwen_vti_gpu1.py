"""Add GPU 1 completion provenance to the audited report folder, then reuse the original exporter."""
import argparse,json,os,shutil
from pathlib import Path

import finalize_qwen_vti_replacement as finalize

ROOT=finalize.ROOT;PARENT=finalize.PARENT
VTI=ROOT/'outputs/qwen-original-vti-1793260357'

def main(output):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    output=output.resolve();assert output.parent==PARENT
    report=finalize.read(output/'report.json');assert report['status']=='verified_complete'
    assert report['analysis_hashes']['auditor']==finalize.sha(ROOT/'scripts/audit_qwen_vti_gpu1_completion.py')
    def copy(src,name):
        dst=output/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert finalize.sha(dst)==finalize.sha(src),str(dst)
        else:shutil.copy2(src,dst)
    for name in ['user-stop-proof-20261029.json','gpu1-amendment.json']:
        copy(VTI/name,'vti-author/metadata/'+name)
    for folder in ['gpu1','gpu1-source-snapshot','gpu1-failed-start-hfcache']:
        for path in (VTI/folder).rglob('*'):
            if path.is_file() and path.suffix in ['.json','.jsonl','.py']:
                copy(path,'vti-author/'+folder+'/'+path.relative_to(VTI/folder).as_posix())
    for script in ['qwen_vti_gpu1.py','launch_qwen_vti_gpu1.py','resume_qwen_vti_gpu1.py','audit_qwen_vti_gpu1_completion.py','export_qwen_vti_gpu1.py']:
        copy(ROOT/'scripts'/script,'analysis-source/'+script)
    copy(VTI/'pope.jsonl','vti-author/pope-gpu0-prefix.jsonl')
    print(json.dumps(finalize.export(PARENT,VTI,output)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
