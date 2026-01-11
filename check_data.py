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
zero_start_count = 0
fixed_zero_start = 0
bug_samples = []

with open(filepath, 'r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    
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

        # Start=0 check
        if start == '0':
            zero_start_count += 1
            if mode == 'clean':
                row['Start'] = '1'
                start = '1'
                fixed_zero_start += 1
        
        # Duplicate check (Fix: check key AFTER potential modifications if relevant, though Start is part of key usually? 
        # Actually in add_data key includes Start. Here let's use the cleaned start if fixed)
        
        # Key definition
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
print(f'Start=0データ: {zero_start_count}')

if mode == 'clean':
    print(f'Start=0修正: {fixed_zero_start}')

if bug_samples:
    print(f'バグサンプル: {bug_samples}')

if mode == 'clean':
    if duplicates > 0 or bugs > 0 or fixed_zero_start > 0:
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean_rows)
        print(f'\n削除・修正完了！保存件数: {len(clean_rows)}件')
    else:
        print('\n変更なし')
elif mode == 'check':
    if duplicates > 0 or bugs > 0 or zero_start_count > 0:
        print('\n修正するには: python check_data.py clean')
