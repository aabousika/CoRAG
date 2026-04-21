"""
Generate answer from search trace
"""
import subprocess

subprocess.run([
    'corag-answer',
    '--trace', 'trace.json',
    '--output', 'answer.md'
])

print("Answer generated ✅")
print("Saved to answer.md")
