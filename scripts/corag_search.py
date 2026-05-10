"""
Search legal database using CoRAG
"""
import subprocess

question = input("Enter your legal question: ")

subprocess.run([
    'corag-search',
    '--question', question,
    '--index-dir', 'data/index',
    '--max-steps', '6',
    '--k', '8',
    '--output', 'trace.json'
])

print("Search complete ")
print("Trace saved to trace.json")
