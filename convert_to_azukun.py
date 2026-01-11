#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
motoデータフォルダのCSVファイルをazukun形式に変換して追加するスクリプト

使用方法:
    python convert_to_azukun.py <入力ファイル>

例:
    python convert_to_azukun.py motoデータ/20251226_duo2.csv
"""

import csv
import re
import sys
import hashlib
from pathlib import Path


def extract_hall_code(hall_url):
    """Hall_URLからホールコードを抽出する（数字のみ）"""
    if not hall_url:
        return None
    # URLからdai_hall_idを抽出
    match = re.search(r'dai_hall_id=(\d+)', hall_url)
    if match:
        return match.group(1)
    # または他のパターン（rack_infoの前のパス部分から）
    # 例: https://bigdaioh.pt.teramoba2.com/bigdaiohhigashi/rack_info
    return None


def generate_id(date_str, hall_url, machine_no):
    """Date, Hall_URL, Machine_Noを使ってIDを生成"""
    # 日付からMM_DDを抽出
    if not date_str:
        return None
    try:
        parts = date_str.split('-')
        if len(parts) == 3:
            month = parts[1].lstrip('0') or '0'
            day = parts[2].lstrip('0') or '0'
        else:
            return None
    except:
        return None
    
    # hall_urlからdai_hall_idを抽出
    hall_code = extract_hall_code(hall_url)
    if not hall_code:
        # URLからホールコードが抽出できない場合、Machine_URLからも試す
        return None
    
    # Machine_No
    if not machine_no:
        return None
    
    return f"{month}_{day}_{hall_code}_{machine_no}"


def generate_id_from_machine_url(date_str, machine_url, machine_no):
    """Machine_URLからdai_hall_idを抽出してIDを生成"""
    if not date_str or not machine_url or not machine_no:
        return None
    
    try:
        parts = date_str.split('-')
        if len(parts) == 3:
            month = parts[1].lstrip('0') or '0'
            day = parts[2].lstrip('0') or '0'
        else:
            return None
    except:
        return None
    
    # Machine_URLからdai_hall_idを抽出
    match = re.search(r'dai_hall_id=(\d+)', machine_url)
    if not match:
        return None
    
    hall_code = match.group(1)
    return f"{month}_{day}_{hall_code}_{machine_no}"


def is_valid_row(row):
    """データの有効性をチェック（Start, Dedama, Statusが存在するか）"""
    start = row.get('Start', '').strip()
    dedama = row.get('Dedama', '').strip()
    status = row.get('Status', '').strip()
    time = row.get('Time', '').strip()
    
    # すべてが空でないことを確認
    if not start or not dedama or not status:
        return False
    
    # StatusがBBまたはRBであることを確認
    if status not in ['BB', 'RB']:
        return False
    
    # StartとDedamaが数値であることを確認
    try:
        int(start)
        int(dedama)
    except ValueError:
        return False
    
    return True


def generate_unique_key(row, row_id):
    """重複チェック用のユニークキーを生成"""
    # ID + Status + Start + Dedama + Time でユニークキーを作成
    return (
        row_id,
        row.get('Status', ''),
        row.get('Start', ''),
        row.get('Dedama', ''),
        row.get('Time', '')
    )


def load_existing_data(output_file):
    """既存のazukunファイルからデータを読み込む"""
    existing_keys = set()
    existing_rows = []
    
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    row.get('ID', ''),
                    row.get('Status', ''),
                    row.get('Start', ''),
                    row.get('Dedama', ''),
                    row.get('Time', '')
                )
                existing_keys.add(key)
                existing_rows.append(row)
    
    return existing_keys, existing_rows


def convert_file(input_file, output_file):
    """ソースファイルをazukun形式に変換"""
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_file}")
        return False
    
    # 既存データを読み込み
    existing_keys, existing_rows = load_existing_data(output_path)
    print(f"既存データ: {len(existing_keys)} 件")
    
    # 統計情報
    stats = {
        'total': 0,
        'invalid': 0,
        'duplicate_source': 0,
        'duplicate_existing': 0,
        'added': 0
    }
    
    new_rows = []
    source_keys = set()  # ソースファイル内での重複チェック用
    
    with open(input_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            stats['total'] += 1
            
            # 有効性チェック（Start, Dedama, Statusが存在するか）
            if not is_valid_row(row):
                stats['invalid'] += 1
                continue
            
            # IDを生成
            date_str = row.get('Date', '')
            machine_url = row.get('Machine_URL', '')
            machine_no = row.get('Machine_No', '')
            
            row_id = generate_id_from_machine_url(date_str, machine_url, machine_no)
            if not row_id:
                stats['invalid'] += 1
                continue
            
            # ユニークキーを生成
            key = generate_unique_key(row, row_id)
            
            # ソースファイル内での重複チェック
            if key in source_keys:
                stats['duplicate_source'] += 1
                continue
            source_keys.add(key)
            
            # 既存データとの重複チェック
            if key in existing_keys:
                stats['duplicate_existing'] += 1
                continue
            
            # 新しい行を追加
            new_row = {
                'ID': row_id,
                'Status': row.get('Status', ''),
                'Start': row.get('Start', ''),
                'Dedama': row.get('Dedama', ''),
                'Time': row.get('Time', '')
            }
            new_rows.append(new_row)
            stats['added'] += 1
    
    # 出力ファイルに書き込み
    all_rows = existing_rows + new_rows
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['ID', 'Status', 'Start', 'Dedama', 'Time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    # 統計を表示
    print(f"\n=== 変換結果 ===")
    print(f"処理行数: {stats['total']}")
    print(f"無効データ（Start/Dedama/Status欠損）: {stats['invalid']}")
    print(f"ソース内重複: {stats['duplicate_source']}")
    print(f"既存データと重複: {stats['duplicate_existing']}")
    print(f"新規追加: {stats['added']}")
    print(f"出力ファイル合計: {len(all_rows)} 件")
    print(f"出力ファイル: {output_path}")
    
    return True


def main():
    if len(sys.argv) < 2:
        print("使用方法: python convert_to_azukun.py <入力ファイル>")
        print("例: python convert_to_azukun.py motoデータ/20251226_duo2.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # 出力ファイルは固定
    output_file = r"C:\Users\ilove\Desktop\解析\20251223_duo2_azukun.csv"
    
    print(f"入力ファイル: {input_file}")
    print(f"出力ファイル: {output_file}")
    
    success = convert_file(input_file, output_file)
    
    if success:
        print("\n変換が完了しました。")
    else:
        print("\n変換に失敗しました。")
        sys.exit(1)


if __name__ == '__main__':
    main()
