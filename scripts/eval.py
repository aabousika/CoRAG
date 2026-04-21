"""
Evaluate CoRAG on legal dataset
"""
import csv
import json
import time
import subprocess
import os

csv.field_size_limit(10000000)

MAX_STEPS = 6
K = 8
NUM_QUESTIONS = 50

# بناء الأسئلة
print(f"Building {NUM_QUESTIONS} evaluation questions...")
questions = []
with open('data/legal_text_classification.csv', 'r') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= NUM_QUESTIONS:
            break
        text_preview = row['case_text'][:200]
        questions.append({
            'case_id': row['case_id'],
            'case_title': row['case_title'],
            'question': f'In the case {row["case_title"]}, what are the specific legal principles, court decisions, and judicial reasoning regarding: {text_preview[:100]}',
            'expected_title': row['case_title']
        })

# تشغيل التقييم
print(f"Running evaluation: max_steps={MAX_STEPS}, k={K}")
print("=" * 50)

results = []
for i, q in enumerate(questions):
    print(f'Evaluating {i+1}/{NUM_QUESTIONS}: {q["question"][:50]}...')
    start = time.time()
    subprocess.run([
        'corag-search',
        '--question', q['question'],
        '--index-dir', 'data/index',
        '--max-steps', str(MAX_STEPS),
        '--k', str(K),
        '--output', f'data/trace_{i}.json'
    ], capture_output=True, text=True)
    latency = time.time() - start

    try:
        with open(f'data/trace_{i}.json') as f:
            trace = json.load(f)
        retrieved_ids = [c.get('doc_id','') for s in trace.get('steps',[]) for c in s.get('chunks',[])]
        retrieved_titles = [c.get('doc_title','') for s in trace.get('steps',[]) for c in s.get('chunks',[])]
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
        results.append({'case_id': q['case_id'], 'hit': False, 'latency': round(latency,2), 'num_steps': 0})

# النتائج
hits = sum(1 for r in results if r['hit'])
total = len(results)
avg_latency = sum(r['latency'] for r in results) / total
avg_steps = sum(r['num_steps'] for r in results) / total

print()
print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"Dataset:      Legal Classification")
print(f"Total:        {total}")
print(f"Hit Rate:     {hits}/{total} = {hits/total*100:.1f}%")
print(f"Avg Steps:    {avg_steps:.2f}")
print(f"Avg Latency:  {avg_latency:.2f}s")
print("=" * 50)

os.makedirs('reports', exist_ok=True)
with open('reports/legal_eval_final.json', 'w') as f:
    json.dump({
        'dataset': 'legal_text_classification',
        'total': total,
        'hits': hits,
        'hit_rate': round(hits/total*100, 1),
        'avg_latency': round(avg_latency, 2),
        'avg_steps': round(avg_steps, 2),
        'details': results
    }, f, indent=2)

print("Results saved to reports/legal_eval_final.json ✅")
