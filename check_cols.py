
import pandas as pd
import pickle

CACHE_FILE = r'C:\Users\ilove\Desktop\解析\chains_cache.pkl'

try:
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    chains_df = cache['chains_df']
    
    print(f"--- chains_df columns ({len(chains_df.columns)}) ---")
    cols = [(i, col) for i, col in enumerate(chains_df.columns)]
    for i, col in cols:
        letter = ""
        n = i + 1
        while n > 0:
            n, r = divmod(n - 1, 26)
            letter = chr(r + 65) + letter
        print(f"{i}: {letter} - {col}")




except Exception as e:
    print(f"Error: {e}")
