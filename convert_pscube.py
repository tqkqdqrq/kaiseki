import csv
import re
import sys
import os
from pathlib import Path

OUTPUT_FILE = r'C:\Users\ilove\Desktop\解析\20251223_duo2_azukun.csv'

def get_date_from_filename(filename):
    """ファイル名から日付(MM_DD)を抽出"""
    # 例: 20260103_pscube.csv -> 1_3
    match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if match:
        month = match.group(2).lstrip('0')
        day = match.group(3).lstrip('0')
        return f"{month}_{day}"
    return None

def is_valid_row(row):
    """
    有効な行か判定
    - 回数, 時刻, ゲーム, ステータス が空でないこと
    - 出玉 が '--' でないこと
    """
    # 必須項目のチェック
    required_cols = ['回数', '時刻', 'ゲーム', 'ステータス']
    for col in required_cols:
        val = row.get(col, '').strip()
        if not val:
            return False
            
    # 出玉チェック
    dedama_str = row.get('出玉', '').strip()
    if dedama_str == '--':
        return False
        
    return True

def convert_pscube_file(input_path):
    input_path = Path(input_path)
    filename = input_path.name
    date_id = get_date_from_filename(filename)
    
    if not date_id:
        print(f"Skipping {filename}: Cannot extract date from filename.")
        return 0, 0

    print(f"Processing {filename} (Date ID: {date_id})...")
    
    # 既存データの読み込み (重複チェック用)
    existing_keys = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get('ID', ''), row.get('Status', ''), row.get('Start', ''), row.get('Dedama', ''), row.get('Time', ''))
                existing_keys.add(key)
    
    new_rows = []
    stats = {'total': 0, 'filtered': 0, 'dup': 0, 'added': 0}
    
    with open(input_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        reversed_rows = list(reversed(rows))
        
        for row in reversed_rows:
            stats['total'] += 1
            
            if not is_valid_row(row):
                stats['filtered'] += 1
                continue
                
            # 各フィールドの取得と変換
            hall_name = row.get('店舗名', '').strip()
            machine_no = row.get('台番号', '').strip()
            
            if not hall_name or not machine_no:
                stats['filtered'] += 1
                continue
                
            # ID 生成: MM_DD_HallName_MachineNo
            row_id = f"{date_id}_{hall_name}_{int(machine_no)}"
            
            # Status 変換
            status_raw = row.get('ステータス', '').strip()
            if status_raw == 'BIG':
                status = 'BB'
            elif status_raw == 'REG':
                status = 'RB'
            else:
                stats['filtered'] += 1 # BB, RB以外も除外（念のため）
                continue
            
            # Start
            try:
                start = str(int(row.get('ゲーム', '').strip()))
            except ValueError:
                stats['filtered'] += 1
                continue
                
            # Dedama
            try:
                dedama = str(int(row.get('出玉', '').strip()))
            except ValueError:
                stats['filtered'] += 1
                continue
                
            # Time
            time_val = row.get('時刻', '').strip()
            
            # 重複チェック
            key = (row_id, status, start, dedama, time_val)
            if key in existing_keys:
                stats['dup'] += 1
                continue
            
            # 新規追加
            new_rows.append({
                'ID': row_id,
                'Status': status,
                'Start': start,
                'Dedama': dedama,
                'Time': time_val
            })
            stats['added'] += 1
            existing_keys.add(key) # 同一ファイル内の重複も防ぐため追加

    # 書き込み (Append mode)
    if new_rows:
        file_exists = os.path.exists(OUTPUT_FILE)
        with open(OUTPUT_FILE, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['ID', 'Status', 'Start', 'Dedama', 'Time'])
            if not file_exists:
                writer.writeheader()
            writer.writerows(new_rows)
            
    print(f"  Result: Total={stats['total']}, Added={stats['added']}, Filtered={stats['filtered']}, Duplicates={stats['dup']}")
    return stats['added'], stats['filtered']

def main():
    target_dir = Path(r'C:\Users\ilove\Desktop\解析\motoデータ\pキューブデータ')
    csv_files = sorted(target_dir.glob('*pscube.csv'))
    
    total_added = 0
    total_filtered = 0
    
    print(f"Target Directory: {target_dir}")
    print(f"Target Files: {len(csv_files)}")
    
    for csv_file in csv_files:
        added, filtered = convert_pscube_file(csv_file)
        total_added += added
        total_filtered += filtered
        
    print(f"\nAll Done. Total Added: {total_added}, Total Filtered: {total_filtered}")

if __name__ == "__main__":
    main()
