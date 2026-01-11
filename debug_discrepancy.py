import pandas as pd
from デュオ解析_v2 import load_data, analyze_chains, calculate_3chain_rule_analysis, INPUT_FILE, HEAVEN_THRESHOLD, COIN_HOLD, BB_GAMES, RB_GAMES

def debug_analysis():
    print("Loading data...")
    df = load_data(INPUT_FILE)
    print(f"Data loaded: {len(df)} rows")
    
    print("Analyzing chains...")
    _, chains_df = analyze_chains(df)
    print(f"Chains generated: {len(chains_df)} rows")
    
    # Run analysis logic (copy of the function logic to inspect internal state if needed, 
    # but here we just call the function if possible, or reimplement lightweight version)
    # Since I need to inspect the 'analysis_data' dictionary which is internal to the function,
    # I cannot just call the function. I need to copy the logic or modify the function to return raw data.
    # I will modify the function in the file to return raw dictionary? 
    # Better: I will copy the logic here to reproduce and debug.
    
    # Re-implement simplified logic to trace discrepancy
    analysis_data = {
        '全体': [],
        '2連目1G_RB': []
    }
    
    chains_df = chains_df.sort_values(by=['ID', 'Chain_Number'])
    
    prev_id = None
    trackers = {cat: {'hits_since': 0, 'first_2chain_idx': None, 'active': False, 'ignore': False} for cat in analysis_data}
    
    diff_found = False
    
    for idx, row in chains_df.iterrows():
        row_id = row['ID']
        chain_len = row['Chain_Length']
        hit_games = row.get('Hit_Games', [])
        hit_statuses = row.get('Hit_Statuses', [])
        
        # Reset
        if row_id != prev_id:
            for cat in trackers:
                trackers[cat]['hits_since'] = 0
                trackers[cat]['first_2chain_idx'] = None
                trackers[cat]['active'] = False
                trackers[cat]['ignore'] = False
            
            if chain_len >= 3:
                for cat in trackers:
                    trackers[cat]['active'] = True
            prev_id = row_id
            continue
            
        # Active Check
        if not trackers['全体']['active']:
            if chain_len >= 3:
                for cat in trackers:
                    trackers[cat]['active'] = True
                    trackers[cat]['hits_since'] = 0
                    trackers[cat]['first_2chain_idx'] = None
                    trackers[cat]['ignore'] = False
            prev_id = row_id
            continue
        
        is_true_heaven = (chain_len >= 3)
        
        # Process logic...
        
        # Capture state BEFORE processing strict match logic to see '0-Through' state
        # For '2連なし' at 'Through 0':
        # Overall and 1G_RB should both have 'first_2chain_idx' as None (if they haven't seen 2-chain yet)
        # and 'ignore' as False.
        
        tr_all = trackers['全体']
        tr_rb = trackers['2連目1G_RB']
        
        # Check specific condition: 0 Through, No 2-Chain
        # If one appends and other doesn't?
        # Only happens if 'ignore' differs.
        
        if tr_all['hits_since'] == 0 and tr_all['first_2chain_idx'] is None:
            # Expect tr_rb to be same
            if tr_rb['ignore']:
                print(f"DISCREPANCY at ID: {row_id}, Chain: {row['Chain_Number']}")
                print(f"Overall: hits={tr_all['hits_since']}, ignore={tr_all['ignore']}")
                print(f"1G_RB: hits={tr_rb['hits_since']}, ignore={tr_rb['ignore']}")
                diff_found = True
                break
        
        # ... logic continues to update state ...
        # (Simplified update logic just to track state changes)
        is_valid_2chain = False
        if chain_len == 2:
            second_hit_g = hit_games[1] if len(hit_games) > 1 else 0
            if second_hit_g < 34:
                is_valid_2chain = True
        
        # Determine match for RB
        match_rb = False
        if is_valid_2chain:
             status_2 = hit_statuses[1] if len(hit_statuses) > 1 else ''
             is_rb = 'RB' in status_2
             is_1g = (hit_games[1] == 1) if len(hit_games) > 1 else False
             if is_rb and is_1g:
                 match_rb = True
        
        # Update Overall
        if not tr_all['ignore']:
             if is_true_heaven:
                 tr_all['hits_since'] = 0
                 tr_all['first_2chain_idx'] = None
                 tr_all['ignore'] = False
             else:
                 if is_valid_2chain and tr_all['first_2chain_idx'] is None:
                     tr_all['first_2chain_idx'] = tr_all['hits_since']
                 
                 inc = 2 if (not is_valid_2chain and chain_len == 2) else 1
                 tr_all['hits_since'] += inc
                 
        # Update RB
        if not tr_rb['ignore']:
             if is_true_heaven:
                 tr_rb['hits_since'] = 0
                 tr_rb['first_2chain_idx'] = None
                 tr_rb['ignore'] = False
             else:
                 if is_valid_2chain and tr_rb['first_2chain_idx'] is None:
                     if match_rb:
                         tr_rb['first_2chain_idx'] = tr_rb['hits_since']
                     else:
                         tr_rb['ignore'] = True
                 
                 inc = 2 if (not is_valid_2chain and chain_len == 2) else 1
                 tr_rb['hits_since'] += inc
                 
        prev_id = row_id

    if not diff_found:
        print("No early discrepancy found in state.")

if __name__ == '__main__':
    debug_analysis()
