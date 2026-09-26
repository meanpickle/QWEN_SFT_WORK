"""Replay the recorded chat protocol; results are written to a new directory."""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    base, tokenizer, output = (x.resolve() for x in (args.base, args.tokenizer, args.output))
    adapters = {name: ROOT / 'outputs' / name for name in ('sft_3k_3ep', 'sft_10k_3ep', 'sft_30k_1ep')}
    for path in [base, tokenizer, *adapters.values()]:
        if not path.is_dir():
            raise SystemExit(f'Missing required directory: {path}')
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'running', 'results': []}
    def save():
        temporary = output / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(output / 'status.json')
    env = dict(os.environ, TOKENIZERS_PARALLELISM='false')
    for task, shots, budget in [('gsm8k', 5, 512), ('mmlu', 5, None), ('ifeval', 0, 1280)]:
        for label, adapter in [('base', None), *adapters.items()]:
            name = f'{task}_{label}'
            model_args = f'pretrained={base},tokenizer={tokenizer},dtype=bfloat16,max_length=4096'
            if adapter:
                model_args += f',peft={adapter}'
            cmd = [sys.executable, '-m', 'lm_eval', '--model', 'hf', '--model_args', model_args,
                   '--tasks', task, '--num_fewshot', str(shots), '--apply_chat_template',
                   '--batch_size', '4', '--device', 'cuda:0', '--seed', '0,1234,1234,1234',
                   '--log_samples', '--output_path', str(output / name)]
            if budget:
                cmd += ['--gen_kwargs', f'max_gen_toks={budget},max_new_tokens={budget},do_sample=False']
            rec = {'name': name, 'command': cmd, 'status': 'running',
                   'started': datetime.datetime.now().isoformat()}
            state['results'].append(rec)
            save()
            with (output / f'{name}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            rec.update(exit_code=result.returncode, status='complete' if result.returncode == 0 else 'failed')
            if result.returncode:
                state['status'] = 'failed'
                save()
                raise SystemExit(result.returncode)
            save()
    state['status'] = 'complete'
    save()

if __name__ == '__main__':
    main()
