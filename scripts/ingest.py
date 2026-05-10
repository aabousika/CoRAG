"""
Ingest legal dataset from CSV to JSONL format
"""
import csv
import json

csv.field_size_limit(10000000)

with open('data/legal_text_classification.csv', 'r') as csv_file, open('data/corpus.jsonl', 'w') as jsonl_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        doc = {
            'id': row['case_id'],
            'title': row['case_title'],
            'text': row['case_text'],
            'metadata': {
                'outcome': row['case_outcome']
            }
        }
        jsonl_file.write(json.dumps(doc) + '\n')

print("Ingestion complete ")
print("Saved to data/corpus.jsonl")
