"""
Build FAISS index from corpus
"""
import subprocess

result = subprocess.run([
    'corag-build-index', 'data/corpus.jsonl', 'data/index',
    '--embedding-model', 'sentence-transformers/msmarco-distilbert-base-v4',
    '--index-type', 'Flat',
    '--chunk-size', '512',
    '--chunk-overlap', '64'
], capture_output=False)

print("Index built successfully ✅")
print("Saved to data/index")
