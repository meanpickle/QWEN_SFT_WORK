"""Reconstruct saved subsets from the exact source revision and row indices."""
import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'data/sampling_manifest.json').read_text(encoding='utf-8-sig'))
    indices = manifest['selected_row_indices']
    assert len(indices) == len(set(indices)) == 30000
    files = sorted(args.source.rglob('train-*-of-00006.parquet'), key=lambda p: p.name)
    assert len(files) == 6 and len({p.name for p in files}) == 6, 'Expected six distinct shards'
    paths = [ROOT / 'data' / f'tulu3_random_{n}.json' for n in (3000, 10000, 30000)]
    assert not any(p.exists() for p in paths), 'Refusing to overwrite existing subsets'
    wanted = set(indices)
    found = {}
    offset = 0
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                if offset in wanted:
                    found[offset] = {k: row[k] for k in ('id', 'messages', 'source') if k in row}
                offset += 1
    assert offset == manifest['total_rows'], 'Source row count mismatch'
    assert len(found) == 30000, 'Missing selected rows'
    ordered = [found[i] for i in indices]
    for n, path in zip((3000, 10000, 30000), paths):
        path.write_text(json.dumps(ordered[:n], ensure_ascii=False), encoding='utf-8')
        print(path, n)

if __name__ == '__main__':
    main()
