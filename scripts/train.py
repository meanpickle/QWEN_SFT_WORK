"""Run one portable LlamaFactory configuration from the repository root."""
import argparse
import subprocess
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    config = args.config.resolve()
    cfg = yaml.safe_load(config.read_text(encoding='utf-8-sig'))
    output = ROOT / cfg['output_dir']
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f'Refusing existing output: {output}')
    subprocess.run(['llamafactory-cli', 'train', str(config)], cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
