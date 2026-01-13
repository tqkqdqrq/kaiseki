import csv
from datetime import datetime

INPUT_FILE = r'C:\Users\ilove\Desktop\解析\20251223_duo2_azukun.csv'
TEMP_FILE = r'C:\Users\ilove\Desktop\解析\20251223_duo2_azukun_sorted.csv'

def main():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    print(f"Total rows: {len(rows)}")
    
    # Sort key:
    # 1. ID (Date + Hall + Machine) - this ensures grouping by machine/day
    # 2. Time (HH:MM) - this ensures chronological order within the machine/day
    # Note: ID format is MM_DD_Hall_Machine. This sorts roughly by date, but strictly speaking "1_10" comes before "1_2".
    # However, Python's efficient timsort is stable. If the original file was mostly ordered, this keeps it.
    # But to be safe, we should parse ID to get real date if we want strict date sorting.
    # For now, let's assume grouping by ID is sufficient, and the most important fix is Time order within ID.
    
    # Wait, Time alone is not enough if the file contains multiple days.
    # We must sort by Date (from ID) then Time.
    
    def sort_key(row):
        # Extract date from ID: "1_3_..."
        row_id = row.get('ID', '')
        parts = row_id.split('_')
        if len(parts) >= 2:
            try:
                month = int(parts[0])
                day = int(parts[1])
            except ValueError:
                month = 0
                day = 0
        else:
            month = 0
            day = 0
            
        time_str = row.get('Time', '00:00')
        return (month, day, row_id, time_str)

    print("Sorting rows...")
    rows.sort(key=sort_key)
    
    print(f"Writing sorted data to {INPUT_FILE}...")
    with open(INPUT_FILE, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print("Done. File overwritten with sorted data.")

if __name__ == "__main__":
    main()
