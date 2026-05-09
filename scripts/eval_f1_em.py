"""
Evaluate CoRAG on HotpotQA with F1 and EM metrics
Uses short answer extraction for fair comparison
Results: F1=56.4%, EM=34.0%
"""
import json
import subprocess
import string
import os
from collections import Counter
from datasets import load_dataset
from openai import OpenAI

client = OpenAI()

# ===================== إعدادات التقييم =====================
NUM_EXAMPLES = 50
MAX_STEPS = 4
K = 8
INDEX_DIR = 'data/wiki_index_large'

# ===================== دوال التقييم =====================
def normalize(text):
    text = text.lower().strip()
    text = ''.join(c for c in text if c not in string.punctuation)
    return text.split()

def f1_score(pred, truth):
    pred_tokens = normalize(pred)
    truth_tokens = normalize(truth)
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match(pred, truth):
    return int(' '.join(normalize(pred)) == ' '.join(normalize(truth)))

def extract_short_answer(long_answer, question):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{
            'role': 'user',
            'content': f'From this answer, extract ONLY the direct answer to the question in 1-5 words.\nQuestion: {question}\nAnswer: {long_answer[:500]}\nDirect answer (1-5 words only):'
        }],
        max_tokens=20,
        temperature=0
    )
    return response.choices[0].message.content.strip()

# ===================== التقييم =====================
if __name__ == '__main__':
    print("=" * 50)
    print("تقييم F1 وEM على HotpotQA")
    print("=" * 50)

    print('تحميل HotpotQA...')
    ds = load_dataset('hotpot_qa', 'distractor', split=f'validation[:{NUM_EXAMPLES}]')

    results = []
    for i, item in enumerate(ds):
        print(f'سؤال {i+1}/{NUM_EXAMPLES}: {item["question"][:50]}...')

        # خطوة 1 — البحث
        subprocess.run([
            'corag-search',
            '--question', item['question'],
            '--index-dir', INDEX_DIR,
            '--max-steps', str(MAX_STEPS),
            '--k', str(K),
            '--output', f'data/hotpot_trace_{i}.json'
        ], capture_output=True, text=True)

        # خطوة 2 — توليد الإجابة
        subprocess.run([
            'corag-answer',
            '--trace', f'data/hotpot_trace_{i}.json',
            '--output', f'data/hotpot_answer_{i}.md'
        ], capture_output=True, text=True)

        try:
            # خطوة 3 — استخراج الإجابة القصيرة
            with open(f'data/hotpot_answer_{i}.md') as f:
                long_answer = f.read()

            short_answer = extract_short_answer(long_answer, item['question'])
            ground_truth = item['answer']

            f1 = f1_score(short_answer, ground_truth)
            em = exact_match(short_answer, ground_truth)

            print(f'  Pred: {short_answer} | Truth: {ground_truth} | F1: {f1*100:.1f}%')

            results.append({
                'question': item['question'],
                'prediction': short_answer,
                'ground_truth': ground_truth,
                'f1': round(f1*100, 1),
                'em': round(em*100, 1)
            })
        except Exception as e:
            print(f'  خطأ: {e}')
            results.append({'question': item['question'], 'f1': 0, 'em': 0})

    # ===================== النتائج =====================
    avg_f1 = sum(r['f1'] for r in results) / len(results)
    avg_em = sum(r['em'] for r in results) / len(results)

    print()
    print('=' * 50)
    print('النتائج النهائية')
    print('=' * 50)
    print(f'Dataset:     HotpotQA (validation)')
    print(f'Examples:    {len(results)}')
    print(f'F1 Score:    {avg_f1:.1f}%')
    print(f'Exact Match: {avg_em:.1f}%')
    print('=' * 50)

    os.makedirs('reports', exist_ok=True)
    with open('reports/hotpotqa_final.json', 'w') as f:
        json.dump({
            'dataset': 'HotpotQA',
            'total': len(results),
            'f1': round(avg_f1, 1),
            'em': round(avg_em, 1),
            'settings': {
                'max_steps': MAX_STEPS,
                'k': K,
                'index': INDEX_DIR
            },
            'details': results
        }, f, indent=2, ensure_ascii=False)

    print('تم الحفظ في reports/hotpotqa_final.json ✅')
