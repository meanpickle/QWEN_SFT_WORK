import json,collections,tarfile,hashlib
from pathlib import Path
r=Path('/workspace/sft-experiment'); dest=r/'reports';dest.mkdir(exist_ok=True)
summary={};audit={}
for root in ['formal_chat_v1','advanced_chat_v1']:
 for job in sorted((r/'evaluation'/root).iterdir()):
  if not job.is_dir():continue
  for f in job.rglob('results*.json'):
   d=json.loads(f.read_text());task=job.name.split('_')[0]
   summary[job.name]={k:v for k,v in d['results'][task].items() if isinstance(v,(int,float)) and 'stderr' not in k}
  if not job.name.startswith('gsm8k'):continue
  groups=collections.defaultdict(dict)
  for f in job.rglob('samples*.jsonl'):
   for line in f.open():
    x=json.loads(line);groups[x['doc_id']][x['filter']]=x
  counts=collections.Counter();examples=[]
  for did,g in groups.items():
   a=g['strict-match'];b=g['flexible-extract'];key=f"strict{int(a['exact_match'])}_flex{int(b['exact_match'])}";counts[key]+=1
   if a['exact_match']!=b['exact_match'] and len(examples)<12:examples.append({'doc_id':did,'target':a['target'],'strict':a['filtered_resps'],'flex':b['filtered_resps'],'response':a['resps'],'category':key})
  audit[job.name]={'counts':dict(counts),'examples':examples};print(job.name,dict(counts),flush=True)
(dest/'metrics.json').write_text(json.dumps(summary,indent=2))
(dest/'gsm8k_extraction_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
lines=['# SFT experiment results','', 'Scores are protocol-specific; all runs use the same chat template.','', 'GSM8K strict extracts the answer after ####; flexible extracts the last number. Neither metric is a superset of the other.','']
for name,values in summary.items():
 lines+=['## '+name]+[f'- {k}: {v:.6f}' for k,v in values.items()]+['']
lines+=['## Extraction audit',json.dumps({k:v['counts'] for k,v in audit.items()},indent=2),'','Archive includes final adapters, configs, sampled data, evaluation evidence and logs. Base model and intermediate checkpoints excluded. This is a server-side archive, not an off-server backup.']
(dest/'results_summary.md').write_text(chr(10).join(lines))
archive=r.parent/'sft_results_backup_20260926.tar.gz'
with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
 for folder in ['configs','data_validated','evaluation','logs','reports','outputs']:
  for f in (r/folder).rglob('*'):
   if not f.is_file() or any(x.startswith('checkpoint-') for x in f.parts):continue
   if folder=='outputs' and f.relative_to(r/'outputs').parts[0] not in ['sft_3k_3ep','sft_10k_3ep','sft_30k_1ep']:continue
   tar.add(f,arcname=str(f.relative_to(r)),recursive=False)
 for f in r.glob('*.py'):tar.add(f,arcname=f.name)
h=hashlib.sha256()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
archive.with_suffix(archive.suffix+'.sha256').write_text(h.hexdigest()+'  '+archive.name+chr(10))
with tarfile.open(archive) as tar:print('ARCHIVE_VERIFIED',len(tar.getnames()),'files',archive.stat().st_size,'bytes',flush=True)
print('DONE',archive,flush=True)
