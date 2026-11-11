"""Export only an audited complete Qwen result, with hashes and no image/weight cache."""
import fcntl,json,os,time,traceback,zipfile
from pathlib import Path
from qwen_gpu0_migration_api import ROOT,sha,save
from audit_qwen_five_profile_suite import actual_job_live

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    out=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386'
    lock=(out/'export.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'pid':os.getpid(),'started_at':time.time(),'exporter_sha256':sha(__file__)}
    def update(**kw):state.update(kw,updated_at=time.time());save(out/'export-status.json',state)
    try:
        while True:
            status=json.loads((out/'finalization-status.json').read_text())
            assert status['stage']!='failed',status
            if status['stage']=='server_verified_complete_pending_local_export':break
            proc=Path('/proc')/str(status['controller_pid'])
            fields=(proc/'stat').read_text().split(') ',1)[1].split()
            assert fields[0] not in ('Z','X') and int(fields[19])==status['controller_start_ticks']
            assert proc.stat().st_uid==os.getuid()
            update(stage='waiting_for_full_official_audit',finalization_stage=status['stage']);time.sleep(30)
        report=json.loads((out/'report.json').read_text())
        assert report['status']=='verified_complete' and report['audit']['all_17500_responses_verified']
        assert sha(out/'report.json')==status['report_sha256']
        jobs=json.loads((out/'all-final-jobs.json').read_text())
        assert all(not actual_job_live(j,out) for j in jobs)
        target=out/'verified-results.zip';assert not target.exists()
        suffixes={'.json','.jsonl','.py','.md','.csv','.txt','.log'}
        excluded={'export-status.json','export-manifest.json','verified-results.zip'}
        paths=[]
        for path in out.rglob('*'):
            if not path.is_file() or path.is_symlink() or path.name in excluded:continue
            relative=path.relative_to(out)
            assert path.resolve().is_relative_to(out.resolve())
            if path.suffix not in suffixes:continue
            paths.append(path)
        for path in [out/'grains-author.pt',out/'artifacts/revis.pt',out/'artifacts/vti.pt']:
            assert path.is_file() and not path.is_symlink();paths.append(path)
        paths=sorted(set(paths),key=lambda p:p.relative_to(out).as_posix())
        hashes={p.relative_to(out).as_posix():sha(p) for p in paths}
        manifest={'status':'verified_complete_export','report_sha256':sha(out/'report.json'),
            'files':hashes,'exporter_sha256':sha(__file__),'all_17500_responses_included':True,
            'raw_ig_and_hidden_arrays_retained_on_server':str(out),
            'cached_model_weights_and_images_not_duplicated':True}
        save(out/'export-manifest.json',manifest)
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipped:
            for p in paths:
                assert sha(p)==hashes[p.relative_to(out).as_posix()]
                zipped.write(p,p.relative_to(out).as_posix())
            zipped.write(out/'export-manifest.json','export-manifest.json')
        update(stage='ready_for_local_download',archive=str(target),archive_sha256=sha(target),
            archive_bytes=target.stat().st_size,report_sha256=sha(out/'report.json'),file_count=len(paths),completed_at=time.time())
    except BaseException:update(stage='failed',error=traceback.format_exc());raise
    finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
