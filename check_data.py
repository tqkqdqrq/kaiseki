#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""azukunファイルの重複・バグデータをチェック＆削除"""

import csv
import sys

filepath = r'C:\Users\ilove\Desktop\解析\20251223_duo2_azukun.csv'

# モード: check or clean
mode = sys.argv[1] if len(sys.argv) > 1 else 'check'

keys = set()
clean_rows = []
duplicates = 0
bugs = 0
bug_samples = []

with open(filepath, 'r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        start = row.get('Start','').strip()
        dedama = row.get('Dedama','').strip()
        status = row.get('Status','').strip()
        
        # Bug check
        is_bug = False
        if not start or not dedama or not status:
            is_bug = True
        elif status not in ['BB', 'RB']:
            is_bug = True
        else:
            try:
                int(start)
                int(dedama)
            except:
                is_bug = True
        
        if is_bug:
            bugs += 1
            if len(bug_samples) < 3:
                bug_samples.append(f"ID={row.get('ID','')}, Status={status}")
            continue
        
        # Duplicate check
        key = (row.get('ID',''), status, start, dedama, row.get('Time',''))
        if key in keys:
            duplicates += 1
            continue
        keys.add(key)
        clean_rows.append(row)

print(f'=== チェック結果 ===')
print(f'クリーンデータ: {len(clean_rows)}')
print(f'重複データ: {duplicates}')
print(f'バグデータ: {bugs}')
if bug_samples:
    print(f'バグサンプル: {bug_samples}')

if mode == 'clean' and (duplicates > 0 or bugs > 0):
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID','Status','Start','Dedama','Time'])
        writer.writeheader()
        writer.writerows(clean_rows)
    print(f'\n削除完了！残り: {len(clean_rows)}件')
elif mode == 'check':
    print('\n削除するには: python check_data.py clean')
