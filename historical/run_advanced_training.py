import os,sys,json,time,subprocess,datetime,fcntl
from pathlib import Path
r=Path('/workspace/sft-experiment')
lock=(r/'logs/advanced_training.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status={'phase':'waiting_for_formal_evaluation','runs':[]}
p=r/'logs/advanced_training_status.json'
def save():
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2));tmp.replace(p)
save()
deadline=time.monotonic()+12*3600
while True:
 try:s=json.loads((r/'evaluation/formal_chat_v1/status.json').read_text())
 except json.JSONDecodeError:time.sleep(5);continue
 if any(x.get('status')=='failed' for x in s.get('results',[])):
  status['phase']='blocked_by_evaluation_failure';save();sys.exit(1)
 if s.get('status')=='complete' and len(s.get('results',[]))==6 and all(x.get('exit_code')==0 for x in s['results']):break
 if time.monotonic()>deadline:
  status['phase']='evaluation_wait_timeout';save();sys.exit(1)
 time.sleep(30)
env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
for name in ['sft_10k_3ep','sft_30k_1ep']:
 out=r/'outputs'/name
 if out.exists() and any(out.iterdir()):
  status.update(phase='stopped_existing_output',output=str(out));save();sys.exit(1)
 cmd=['llamafactory-cli','train',str(r/'configs'/f'{name}.yaml')]
 rec={'name':name,'started':datetime.datetime.now().isoformat(),'status':'running','command':cmd};status['runs'].append(rec);status['phase']='training';save()
 print('START',name,flush=True)
 with (r/'logs'/f'{name}.log').open('w') as f:result=subprocess.run(cmd,cwd='/workspace/LlamaFactory',env=env,stdout=f,stderr=subprocess.STDOUT)
 rec.update(exit_code=result.returncode,ended=datetime.datetime.now().isoformat(),status='complete' if result.returncode==0 else 'failed');save()
 if result.returncode:status['phase']='training_failed';save();sys.exit(result.returncode)
 assert (out/'adapter_model.safetensors').is_file(),'Adapter missing'
 print('END',name,flush=True)
status['phase']='training_complete_evaluation_pending';save()
