import os, sys, json, subprocess, datetime
from pathlib import Path
root=Path('/workspace/sft-experiment/evaluation/formal_chat_v1')
root.mkdir(exist_ok=True)
base='/workspace/models/Qwen3-4B-Base'
adapter='/workspace/sft-experiment/outputs/sft_3k_3ep'
env=os.environ.copy()
env.update(HF_ENDPOINT='https://hf-mirror.com',HF_HUB_OFFLINE='0',HF_DATASETS_OFFLINE='0',HF_HUB_ETAG_TIMEOUT='30',HF_HUB_DOWNLOAD_TIMEOUT='60',TOKENIZERS_PARALLELISM='false')
manifest={'protocol':'same saved SFT tokenizer and chat template for both models; no test subset limit','models':{'base':base,'sft':adapter},'tasks':{'gsm8k':{'fewshot':5,'max_new_tokens':512},'mmlu':{'fewshot':5},'ifeval':{'fewshot':0,'max_new_tokens':1280}},'batch_size':4,'max_length':4096,'seed':'0,1234,1234,1234','results':[]}
def save():
 (root/'status.json').write_text(json.dumps(manifest,indent=2))
save()
with (root/'environment.txt').open('w') as f: subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
for task in ['gsm8k','mmlu','ifeval']:
 for label in ['base','sft']:
  name=f'{task}_{label}'
  out=root/name
  if out.exists(): raise RuntimeError(f'Refusing to overwrite {out}')
  args=f'pretrained={base},tokenizer={adapter},dtype=bfloat16,max_length=4096'
  if label=='sft': args+=f',peft={adapter}'
  cmd=[sys.executable,'-m','lm_eval','--model','hf','--model_args',args,'--tasks',task,'--num_fewshot',str(manifest['tasks'][task]['fewshot']),'--apply_chat_template','--batch_size','4','--device','cuda:0','--seed','0,1234,1234,1234','--log_samples','--output_path',str(out)]
  if task!='mmlu':
   budget=manifest['tasks'][task]['max_new_tokens']
   cmd+=['--gen_kwargs',f'max_gen_toks={budget},max_new_tokens={budget},do_sample=False']
  rec={'name':name,'command':cmd,'started':datetime.datetime.now().isoformat(),'status':'running'}
  manifest['results'].append(rec);save()
  print('START',name,flush=True)
  with (root/f'{name}.log').open('w') as f:
   result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  rec.update(exit_code=result.returncode,status='complete' if result.returncode==0 else 'failed',ended=datetime.datetime.now().isoformat());save()
  print('END',name,result.returncode,flush=True)
  if result.returncode: sys.exit(result.returncode)
manifest['status']='complete';save()
print('ALL_COMPARISONS_COMPLETE',flush=True)
