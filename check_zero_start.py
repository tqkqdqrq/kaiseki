import csv

filepath = r'C:\Users\ilove\Desktop\解析\20251223_duo2_azukun.csv'

count = 0
samples = []

with open(filepath, 'r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        start_val = row.get('Start', '').strip()
        # Check if it looks like 0
        if start_val == '0':
            count += 1
            if len(samples) < 5:
                samples.append(row)

print(f"Start=0 count: {count}")
if count > 0:
    print("Samples:")
    for s in samples:
        print(s)
