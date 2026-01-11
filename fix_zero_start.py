import csv
import shutil

filepath = r'C:\Users\ilove\Desktop\解析\20251223_duo2_azukun.csv'
temp_filepath = filepath + '.tmp'

count = 0
updated_rows = []

with open(filepath, 'r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        if row.get('Start', '').strip() == '0':
            row['Start'] = '1'
            count += 1
        updated_rows.append(row)

if count > 0:
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    print(f"Updated {count} records where Start=0 to Start=1.")
else:
    print("No records found with Start=0.")
