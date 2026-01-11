import sys

file_path = '軽量化デュオ解析_v2.py'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 1-based line numbers to remove: 1700 to 1821 (inclusive)
# 0-based indices: 1699 to 1821 (slice end exclusive)
start_idx = 1699
end_idx = 1821

# Validate content
# Line 1700 should contain col_start definition
# Line 1822 should be if __name__ == "__main__":

if len(lines) < 1822:
    print(f"Error: File has only {len(lines)} lines.")
    sys.exit(1)

line_start = lines[start_idx]
line_next = lines[end_idx]

print(f"Line 1700: {line_start.rstrip()}")
print(f"Line 1822: {line_next.rstrip()}")

if "col_start = col_map.get('初当G')" in line_start and 'if __name__ == "__main__":' in line_next:
    print("Pattern matched. Deleting lines...")
    new_lines = lines[:start_idx] + lines[end_idx:]
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Done.")
else:
    print("Pattern mismatch! Aborting.")
