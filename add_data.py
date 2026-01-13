#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""20251226_duo2.csvを20251223_duo2_azukun.csvに追加"""

import csv

input_file = r'C:\Users\ilove\Desktop\解析\motoデータ\20260112_duo2.csv'
output_file = r'C:\Users\ilove\Desktop\解析\20251223_duo2_azukun.csv'

# 既存データ読み込み
existing_keys = set()
existing_rows = []
with open(output_file, 'r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row.get('ID',''), row.get('Status',''), row.get('Start',''), row.get('Dedama',''), row.get('Time',''))
        existing_keys.add(key)
        existing_rows.append(row)
print(f'既存データ: {len(existing_rows)} 件')

# 新規データ読み込み
stats = {'total':0, 'invalid':0, 'dup_src':0, 'dup_exist':0, 'added':0}
new_rows = []
source_keys = set()

with open(input_file, 'r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        stats['total'] += 1
        start = row.get('Start','').strip()
        dedama = row.get('Dedama','').strip()
        status = row.get('Status','').strip()
        if not start or not dedama or not status:
            stats['invalid'] += 1
            continue
        if status not in ['BB','RB']:
            stats['invalid'] += 1
            continue
        try:
            int(start)
            int(dedama)
        except:
            stats['invalid'] += 1
            continue
        
        date_str = row.get('Date','')
        hall_name = row.get('Hall_Name','')
        machine_no = row.get('Machine_No','')
        if not date_str or not hall_name or not machine_no:
            stats['invalid'] += 1
            continue
        parts = date_str.split('-')
        if len(parts) == 3:
            month = parts[1].lstrip('0') or '0'
            day = parts[2].lstrip('0') or '0'
        else:
            stats['invalid'] += 1
            continue
        row_id = f'{month}_{day}_{hall_name}_{machine_no}'
        
        key = (row_id, status, start, dedama, row.get('Time',''))
        if key in source_keys:
            stats['dup_src'] += 1
            continue
        source_keys.add(key)
        if key in existing_keys:
            stats['dup_exist'] += 1
            continue
        
        new_rows.append({'ID':row_id, 'Status':status, 'Start':start, 'Dedama':dedama, 'Time':row.get('Time','')})
        stats['added'] += 1

# 書き込み
all_rows = existing_rows + new_rows
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['ID','Status','Start','Dedama','Time'])
    writer.writeheader()
    writer.writerows(all_rows)

print(f'処理行数: {stats["total"]}')
print(f'無効データ: {stats["invalid"]}')
print(f'ソース内重複: {stats["dup_src"]}')
print(f'既存と重複: {stats["dup_exist"]}')
print(f'新規追加: {stats["added"]}')
print(f'合計: {len(all_rows)} 件')
print('完了しました')
