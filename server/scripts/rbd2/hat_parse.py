"""ALOHa object parsing into the shared cache (hat_common.py); only texts not yet cached are parsed.
The official parser is gone: OpenAI shut down gpt-3.5-turbo, gpt-4 and gpt-4-turbo on 2026-10-23. Backends:
  hf:CHECKPOINT   local LLM with the official prompts (system prompt, first 3 in-context examples, as METRICS['aloha']), greedy.
                  Run with .venv on one GPU.
  openai:MODEL    official OpenAIObjectParser with another model, e.g. openai:gpt-4o. Run with venvs/aloha; OPENAI_API_KEY.
  spacy           official SpacyObjectParser (nouns, no LLM). Run with venvs/aloha.
usage: hat_parse.py PARSER [--hat] [--run RUNDIR ...] [--batch 8]
  always the references of sets/hat385; --hat also HAT's own 400 captions; --run also every caption in RUNDIR's units."""
import argparse,concurrent.futures as cf,glob,importlib.util,json,time
from hat_common import HAT_JSON,L,SET,key,load_cache,parser_file,read

def worklist(a):
    w={}
    def add(kind,text,refs=None):w[key(kind,text)]=(kind,text,refs)
    for r in read(SET):add('multi','\n'.join(r['metadata']['references']),r['metadata']['references'])
    if a.hat:
        for s in json.loads(HAT_JSON.read_text(encoding='utf-8')):add('single',s['caption'])
    for d in a.run:
        for f in glob.glob(f'{d}/*.jsonl'):
            for r in read(f):add('single',r['prediction'])
    return w

def snapshot(checkpoint):
    """The one cached snapshot: models fetched at a fixed revision have no refs/main, which offline loading by name needs."""
    s=glob.glob(str(L/'.cache/huggingface/hub'/f"models--{checkpoint.replace('/','--')}"/'snapshots/*'))
    return s[0] if len(s)==1 else checkpoint

def hf_backend(checkpoint):
    import torch
    from transformers import AutoModelForImageTextToText,AutoProcessor
    checkpoint=snapshot(checkpoint);print('parser model',checkpoint,flush=True)
    spec=importlib.util.spec_from_file_location('aloha_prompts',L/'bench/aloha/src/aloha/object_parser/prompts.py')
    P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
    proc=AutoProcessor.from_pretrained(checkpoint);tok=proc.tokenizer;tok.padding_side='left'
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    m=AutoModelForImageTextToText.from_pretrained(checkpoint,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(dev).eval()
    def msgs(kind,text,refs):
        if kind=='single':sysp,ex,q=P.SINGLE_OBJECT_SYSTEM_PROMPT,P.SINGLE_OBJECT_EXAMPLES,f'Caption: {text}\nObjects:'
        else:sysp,ex,q=P.MULTI_OBJECT_SYSTEM_PROMPT,P.MULTI_OBJECT_EXAMPLES,'Captions:\n'+'\n'.join(f'- {c}' for c in refs)+'\nObjects:'
        return [{'role':'system','content':sysp}]+[{'role':r,'content':c} for u,s in ex[:3] for r,c in (('user',u),('assistant',s))]+[{'role':'user','content':q}]
    def gen(batch):
        texts=[proc.apply_chat_template(msgs(*b),tokenize=False,add_generation_prompt=True) for b in batch]
        x=proc(text=texts,padding=True,return_tensors='pt').to(dev)
        with torch.no_grad():o=m.generate(**x,do_sample=False,num_beams=1,temperature=None,top_p=None,top_k=None,repetition_penalty=1.0,max_new_tokens=512,use_cache=True)
        return tok.batch_decode(o[:,x['input_ids'].shape[1]:],skip_special_tokens=True)
    return gen

def official_backend(parser):
    if parser=='spacy':
        from aloha.object_parser import SpacyObjectParser;p=SpacyObjectParser(3,3)
    else:
        from aloha.object_parser.openai import OpenAIObjectParser;p=OpenAIObjectParser(3,3,model=parser.split(':',1)[1])
    def one(b):
        for k in range(5):
            try:return p.extract_objects_single_caption(b[1]) if b[0]=='single' else p.extract_objects_multiple_captions(b[2])
            except Exception:
                if k==4:raise
                time.sleep(2**k)
    ex=cf.ThreadPoolExecutor(1 if parser=='spacy' else 8)
    return lambda batch:list(ex.map(one,batch))

def split_on_oom(gen,part):
    """A batch of long captions can overflow an A5000 next to the model; halve it instead of failing the night's run."""
    try:return gen(part)
    except RuntimeError as e:
        if 'out of memory' not in str(e).lower() or len(part)==1:raise
        import torch;torch.cuda.empty_cache();m=len(part)//2;print(f'out of memory on {len(part)} texts, splitting',flush=True)
        return split_on_oom(gen,part[:m])+split_on_oom(gen,part[m:])

def main(a):
    w=worklist(a);have=load_cache(a.parser);todo=[(k,v) for k,v in w.items() if k not in have]
    print(f'{a.parser}: {len(w)} texts, {len(todo)} to parse',flush=True)
    f=parser_file(a.parser);f.parent.mkdir(parents=True,exist_ok=True)
    with f.open('a',encoding='utf-8') as out:
        for k,v in todo:
            if not v[1].strip():out.write(json.dumps({'key':k,'raw':''})+'\n')
        todo=[x for x in todo if x[1][1].strip()]
        if not todo:return
        gen=hf_backend(a.parser[3:]) if a.parser.startswith('hf:') else official_backend(a.parser)
        todo.sort(key=lambda x:len(x[1][1]));t0=time.time()
        for i in range(0,len(todo),a.batch):
            part=todo[i:i+a.batch]
            for (k,_),raw in zip(part,split_on_oom(gen,[v for _,v in part])):out.write(json.dumps({'key':k,'raw':raw})+'\n')
            out.flush();print(f'{i+len(part)}/{len(todo)} {time.time()-t0:.0f}s',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parser');p.add_argument('--hat',action='store_true')
    p.add_argument('--run',action='append',default=[]);p.add_argument('--batch',type=int,default=8);main(p.parse_args())
