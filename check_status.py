import csv
from collections import Counter

filepath = r'C:\Users\ilove\Desktop\解析\20251223_duo2_azukun.csv'

status_counter = Counter()
irregular_samples = []

with open(filepath, 'r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        status = row.get('Status', '').strip()
        status_counter[status] += 1
        
        if status not in ['BB', 'RB']:
            if len(irregular_samples) < 5:
                irregular_samples.append(row)

print("Status Counts:")
for s, c in status_counter.items():
    print(f"  {s}: {c}")

irregular_count = sum(c for s, c in status_counter.items() if s not in ['BB', 'RB'])
print(f"\nIrregular Status Count: {irregular_count}")

if irregular_count > 0:
    print("Samples:")
    for s in irregular_samples:
        print(s)
