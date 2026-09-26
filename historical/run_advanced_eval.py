import os,json,subprocess,datetime,fcntl
from pathlib import Path
r=Path('/workspace/sft-experiment')
e=r/'evaluation'
lock=(e/'advanced_eval.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old=json.loads((e/'formal_chat_v1/status.json').read_text())
assert old['status']=='complete'
train=json.loads((r/'logs/advanced_training_status.json').read_text())
assert train['phase']=='training_complete_evaluation_pending'
out=e/'advanced_chat_v1'
out.mkdir(exist_ok=False)
s={'status':'running','results':[]}
def save():
 p=out/'status.tmp'
 p.write_text(json.dumps(s,indent=2));p.replace(out/'status.json')
base_adapter=str(r/'outputs/sft_3k_3ep')
env=os.environ.copy()
env.update(HF_ENDPOINT='https://hf-mirror.com',HF_HUB_OFFLINE='0',HF_DATASETS_OFFLINE='0',HF_HUB_ETAG_TIMEOUT='30',HF_HUB_DOWNLOAD_TIMEOUT='60',TOKENIZERS_PARALLELISM='false')
for task in ['gsm8k','mmlu','ifeval']:
 source=next(x for x in old['results'] if x['name']==task+'_sft')
 for model in ['sft_10k_3ep','sft_30k_1ep']:
  adapter=r/'outputs'/model
  assert (adapter/'adapter_model.safetensors').stat().st_size>0
  name=task+'_'+model
  cmd=list(source['command'])
  i=cmd.index('--model_args')+1
  assert 'peft='+base_adapter in cmd[i]
  cmd[i]=cmd[i].replace('peft='+base_adapter,'peft='+str(adapter))
  cmd[cmd.index('--output_path')+1]=str(out/name)
  rec={'name':name,'command':cmd,'status':'running','started':datetime.datetime.now().isoformat()}
  s['results'].append(rec);save()
  print('START',name,flush=True)
  with (out/(name+'.log')).open('w') as log:
   result=subprocess.run(cmd,cwd='/workspace/LlamaFactory',env=env,stdout=log,stderr=subprocess.STDOUT)
  rec.update(exit_code=result.returncode,status='complete' if result.returncode==0 else 'failed',ended=datetime.datetime.now().isoformat())
  save()
  if result.returncode:s['status']='failed';save();raise SystemExit(result.returncode)
  print('DONE',name,flush=True)
s['status']='complete';save()
