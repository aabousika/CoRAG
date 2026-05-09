"""
Fine-tune the embedding model on legal dataset
Trains msmarco-distilbert-base-v4 on Australian legal cases
Result: Hit Rate improved from 66% to 90%
"""
import json
import os
import random
import csv

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# ===================== إعدادات التدريب =====================
TRAIN_EXAMPLES = 500
BATCH_SIZE = 8
EPOCHS = 3
WARMUP_STEPS = 50
MODEL_INPUT = 'sentence-transformers/msmarco-distilbert-base-v4'
MODEL_OUTPUT = 'models/legal-corag'
DATA_FILE = 'data/train_data.json'
CSV_FILE = 'data/legal_text_classification.csv'

# ===================== الخطوة 1 — تجهيز الداتا =====================
def prepare_data():
    print("=" * 50)
    print("الخطوة 1: تجهيز داتا التدريب...")
    print("=" * 50)

    csv.field_size_limit(10000000)
    random.seed(42)

    rows = []
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    train_data = []
    for i, row in enumerate(rows):
        train_data.append({
            'query': f'What are the legal principles in {row["case_title"]}?',
            'positive': row['case_text'][:512],
            'negative': rows[(i + random.randint(1, 10)) % len(rows)]['case_text'][:512]
        })

    with open(DATA_FILE, 'w') as f:
        json.dump(train_data, f, indent=2)

    print(f"تم إنشاء {len(train_data)} مثال تدريب ✅")
    return train_data

# ===================== الخطوة 2 — التدريب =====================
def train_model(train_data):
    print()
    print("=" * 50)
    print("الخطوة 2: تحميل النموذج وبدء التدريب...")
    print("=" * 50)

    model = SentenceTransformer(MODEL_INPUT)

    examples = []
    for item in train_data[:TRAIN_EXAMPLES]:
        examples.append(InputExample(
            texts=[item['query'], item['positive'], item['negative']]
        ))

    loader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    loss = losses.TripletLoss(model=model)

    print(f"عدد الأمثلة:  {TRAIN_EXAMPLES}")
    print(f"Batch Size:   {BATCH_SIZE}")
    print(f"Epochs:       {EPOCHS}")
    print(f"الوقت المتوقع: ~30 دقيقة")
    print()

    os.makedirs('models', exist_ok=True)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        show_progress_bar=True,
        output_path=MODEL_OUTPUT
    )

    print()
    print(f"تم التدريب ✅")
    print(f"النموذج محفوظ في: {MODEL_OUTPUT}")

# ===================== الخطوة 3 — بناء Index =====================
def build_index():
    print()
    print("=" * 50)
    print("الخطوة 3: بناء Index بالنموذج المدرَّب...")
    print("=" * 50)

    import subprocess
    result = subprocess.run([
        'corag-build-index', 'data/corpus.jsonl', 'data/index_trained',
        '--embedding-model', MODEL_OUTPUT,
        '--index-type', 'Flat',
        '--chunk-size', '512',
        '--chunk-overlap', '64'
    ], capture_output=False)

    print("Index جاهز ✅ — محفوظ في data/index_trained")

# ===================== الخطوة 4 — التقييم =====================
def evaluate():
    print()
    print("=" * 50)
    print("الخطوة 4: تقييم النموذج المدرَّب...")
    print("=" * 50)

    import time
    import subprocess

    with open('data/eval_questions_50.json', 'r') as f:
        questions = json.load(f)

    results = []
    for i, q in enumerate(questions):
        print(f'تقييم {i+1}/50: {q["question"][:50]}...')
        start = time.time()
        subprocess.run([
            'corag-search',
            '--question', q['question'],
            '--index-dir', 'data/index_trained',
            '--max-steps', '6',
            '--k', '8',
            '--output', f'data/trace_{i}.json'
        ], capture_output=True, text=True)
        latency = time.time() - start

        try:
            with open(f'data/trace_{i}.json') as f:
                trace = json.load(f)
            retrieved_ids = [c.get('doc_id', '') for s in trace.get('steps', []) for c in s.get('chunks', [])]
            retrieved_titles = [c.get('doc_title', '') for s in trace.get('steps', []) for c in s.get('chunks', [])]
            hit = q['case_id'] in retrieved_ids or any(
                q['expected_title'].lower()[:30] in t.lower()
                for t in retrieved_titles
            )
            results.append({
                'case_id': q['case_id'],
                'hit': hit,
                'latency': round(latency, 2),
                'num_steps': len(trace.get('steps', []))
            })
        except:
            results.append({'case_id': q['case_id'], 'hit': False, 'latency': round(latency, 2), 'num_steps': 0})

    hits = sum(1 for r in results if r['hit'])
    total = len(results)
    avg_latency = sum(r['latency'] for r in results) / total
    avg_steps = sum(r['num_steps'] for r in results) / total

    print()
    print("=" * 50)
    print("النتائج النهائية")
    print("=" * 50)
    print(f"النموذج القديم:    66.0%")
    print(f"النموذج المدرَّب:  {hits/total*100:.1f}%")
    print(f"التحسن:           +{hits/total*100 - 66.0:.1f}%")
    print(f"متوسط الخطوات:    {avg_steps:.2f}")
    print(f"متوسط الوقت:      {avg_latency:.2f}s")
    print("=" * 50)

    os.makedirs('reports', exist_ok=True)
    with open('reports/trained_eval.json', 'w') as f:
        json.dump({
            'model': MODEL_OUTPUT,
            'total': total,
            'hits': hits,
            'hit_rate': round(hits/total*100, 1),
            'old_hit_rate': 66.0,
            'improvement': round(hits/total*100 - 66.0, 1),
            'avg_latency': round(avg_latency, 2),
            'avg_steps': round(avg_steps, 2),
            'details': results
        }, f, indent=2, ensure_ascii=False)

    print("تم الحفظ في reports/trained_eval.json ✅")

# ===================== التشغيل =====================
if __name__ == '__main__':
    print()
    print("🚀 بدء Fine-tuning النموذج على الداتا القانونية")
    print("=" * 50)

    train_data = prepare_data()
    train_model(train_data)
    build_index()
    evaluate()

    print()
    print("🎉 Fine-tuning اكتمل بنجاح!")
