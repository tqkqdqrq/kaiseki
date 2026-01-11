import csv
from collections import defaultdict
import statistics
import re

# File path
FILE_PATH = r'C:\Users\ilove\Desktop\解析\20251223_duo2.csv'

# Constants
COIN_HOLD = 35.3  # G per 50 coins
HEAVEN_THRESHOLD = 33  # G

def parse_int(s):
    try:
        return int(float(s) if s else 0)
    except ValueError:
        return 0

def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader) # Skip header
        # Header expected: Hall_Name, Hall_URL, Machine_No, ...
        # Based on view_file:
        # 0: Hall_Name
        # 1: Hall_URL
        # 2: Machine_No
        # 3: Machine_Name_Used
        # 4: Machine_URL (contains date)
        # 5: Count (Hit number)
        # 6: Time
        # 7: Start (Games)
        # 8: Dedama (Payout)
        # 9: Status (Bonus Type)
        
        for row in reader:
            if len(row) < 10: continue
            
            machine_no = row[2]
            try:
                count = int(row[5])
                games = int(row[7])
                dedama = int(row[8])
            except ValueError:
                continue
                
            # Extract date from URL
            date_match = re.search(r'target_date=(\d{4}-\d{2}-\d{2})', row[4])
            date = date_match.group(1) if date_match else "Unknown"
            
            status = row[9]
            
            data.append({
                'hall': row[0],
                'date': date,
                'machine': machine_no,
                'count': count,
                'games': games,
                'dedama': dedama,
                'status': status
            })
    return data

def analyze_machines(data):
    # Group by (Hall, Date, Machine)
    machines = defaultdict(list)
    for row in data:
        key = (row['hall'], row['date'], row['machine'])
        machines[key].append(row)
    
    # Sort hits by 'count' for each machine
    for key in machines:
        machines[key].sort(key=lambda x: x['count'])
        
    return machines

def calculate_metrics(machines):
    morning_initial_games = []
    morning_heaven_chains = [] # List of chains (list of hits) that started in morning and went to heaven
    all_chains = [] # List of all chains (list of hits)
    morning_through_counts = [] # Number of singles before heaven
    
    # Stats storage
    morning_chain_payouts = []
    morning_chain_lengths = []
    
    # Transition counts for continuation rate
    # Dict mapping current_len -> count_that_continued
    continuation_counts = defaultdict(int)
    continuation_totals = defaultdict(int)

    for machine_key, hits in machines.items():
        if not hits: continue
        
        # 1. Identify Chains
        chains = []
        current_chain = []
        
        # First hit is always new chain (Morning)
        current_chain.append(hits[0])
        
        for i in range(1, len(hits)):
            hit = hits[i]
            if hit['games'] <= HEAVEN_THRESHOLD:
                # Continuation (Heaven)
                current_chain.append(hit)
            else:
                # New Chain (Initial Hit)
                chains.append(current_chain)
                current_chain = [hit]
        
        # Append the last chain
        if current_chain:
            chains.append(current_chain)
        
        # Add to all_chains
        all_chains.extend(chains)
        
        # 2. Morning Analysis (Chain 0)
        morning_chain = chains[0]
        
        # Morning Initial Games (Hit 1 of Chain 1)
        morning_initial_games.append(morning_chain[0]['games'])
        
        # Check if morning hit went to heaven (Length >= 2)
        is_heaven = len(morning_chain) >= 2
        if is_heaven:
            morning_heaven_chains.append(morning_chain)
            morning_chain_lengths.append(len(morning_chain))
            
            # Payout Calc
            # Net = Sum(Dedama - (Games/35.3)*50)
            net_payout = 0
            for h in morning_chain:
                cost = (h['games'] / COIN_HOLD) * 50
                net_payout += (h['dedama'] - cost)
            morning_chain_payouts.append(net_payout)
            
        # 3. Through Analysis
        # Count consecutive chains of length 1 starting from index 0
        through_count = 0
        found_heaven = False
        for ch in chains:
            if len(ch) == 1:
                through_count += 1
            else:
                # Found a heaven chain
                found_heaven = True
                break
        
        # If we went through whole day without heaven, do we count it?
        # User asked "Max count until entering Heaven". 
        # If it never entered, it's strictly correct to say "It survived X times without entering".
        # But for "Until entering", usually implies we only count those that eventually entered or count the max observed streak.
        # I will store the streak regardless. If it ended the day, it's a "streak of X".
        # However, to be precise on "Entering Heaven", maybe separate?
        # User phrase: "朝一当たらなくてその後スルーしたときの天国に入るまでの最大回数"
        # I will just track the streak of singles from start.
        # Note: If Morning Chain was Heaven (Length > 1), Through Count is 0. 
        # Wait, if Morning Chain Length=1, Count=1.
        # If Chain 1 (Len=1) -> Chain 2 (Len=2). Count = 1 through.
        # Logic: Count chains where len=1, processing in order, stop at first len>=2.
        
        # Re-eval logic:
        # If chains[0] len >= 2: Through = 0.
        # If chains[0] len=1, chains[1] len=1, chains[2] len=2: Through = 2.
        
        t_seq = 0
        entered = False
        for ch in chains:
            if len(ch) >= 2:
                entered = True
                break
            t_seq += 1
        
        # Use t_seq. (If never entered, t_seq = total chains).
        morning_through_counts.append(t_seq)
        
        # 4. Continuation Rates (Morning Chains Only? User said "Morning Info... And continuation rates".
        # Usually continuation is a general spec. I will calculate for ALL chains that are "Heaven" (Length>=2).
        # Or should I calc for ALL chains including singles?
        # "2連の次は3連に行っている台" means "Given a chain reached 2, did it reach 3?"
        # Only chains with length >= 2 are relevant for "2->3".
        # Even a length 1 chain is relevant for "1->2" (Heaven Transition).
        
        # Let's do it for ALL chains to get better stats.
        for ch in chains: # Use all chains or morning_heaven_chains? Using all_chains gives better 'Mode' stats.
            # But wait, Initial Hit -> Heaven is 1->2. 
            # If Chain Length is 1, it failed 1->2.
            # If Chain Length is 2, it succeeded 1->2, failed 2->3.
            length = len(ch)
            for l in range(1, length + 1):
                # If we are at length l, it means we reached l.
                # So we passed l-1 -> l check.
                # Actually simpler:
                # A chain of length L contributes to "reached 1, 2, ..., L".
                # It failed at L+1.
                pass
            
            # Better structure:
            # Total chains reaching N: Count(LEN >= N)
            # Probability N -> N+1: Count(LEN >= N+1) / Count(LEN >= N)
            
            # We will compute this later from list of lengths.
            pass

    # -- Summary Calculations --
    
    # 1. Morning Initial Games
    avg_morning_start = statistics.mean(morning_initial_games) if morning_initial_games else 0
    
    # 2. Heaven Transition Rate (Morning)
    # Count(Morning Length >= 2) / Total Morning Chains
    total_morning = len(machines)
    header_transition_count = sum(1 for ml in morning_through_counts if ml == 0) # 0 throughs means Morning was Heaven
    # Wait, using through_counts logic: if morning was heaven, t_seq=0. Correct.
    # But wait, purely "Hit 2 <= 33G".
    # Yes, that defines Length >= 2.
    heaven_rate = (header_transition_count / total_morning * 100) if total_morning else 0
    
    # 3. Avg Heaven Chain Count (Morning)
    # Average length of chains that entered heaven (Length >= 2)
    avg_chain_len = statistics.mean(morning_chain_lengths) if morning_chain_lengths else 0
    
    # 4. Avg Chain Payout (Morning Heaven)
    avg_chain_payout = statistics.mean(morning_chain_payouts) if morning_chain_payouts else 0
    
    # 5. Through Analysis
    # Max count
    max_through = max(morning_through_counts) if morning_through_counts else 0
    # Avg count (excluding immediate heavens? or including?)
    # "朝一当たらなくて...". This implies conditional on not hitting immediately?
    # Usually "Average Throughs" includes the 0s? No, "Average Resets to Heaven".
    # User asks: "Max times...". I'll provide Max and Distribution.
    
    # 6. Continuation Rates (All Chains)
    # Gather all lengths
    all_lengths = [len(ch) for ch in all_chains]
    max_len_all = max(all_lengths) if all_lengths else 0
    
    continuation_stats = []
    for i in range(1, max_len_all + 1):
        reached_i = sum(1 for l in all_lengths if l >= i)
        reached_next = sum(1 for l in all_lengths if l >= i+1)
        
        if reached_i == 0:
            rate = 0.0
        else:
            rate = (reached_next / reached_i) * 100
        
        continuation_stats.append({
            'current': i,
            'next': i+1,
            'count_reached': reached_i,
            'count_next': reached_next,
            'rate': rate
        })
        
    return {
        'total_machines': total_morning,
        'avg_morning_start': avg_morning_start,
        'morning_heaven_rate': heaven_rate,
        'avg_morning_heaven_len': avg_chain_len,
        'avg_morning_payout': avg_chain_payout,
        'max_morning_through': max_through, 
        'continuation_stats': continuation_stats
    }

def print_results(results):
    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(f"--- Morning Analysis (Total Machines: {results['total_machines']}) ---\n")
        f.write(f"Average Morning Initial Hit Games: {results['avg_morning_start']:.1f} G\n")
        f.write(f"Morning Heaven Transition Rate: {results['morning_heaven_rate']:.1f}%\n")
        f.write(f"Average Heaven Chain Length (Morning): {results['avg_morning_heaven_len']:.2f} hits (Including Initial)\n")
        f.write(f"Average Chain Net Payout (Morning Heaven): {results['avg_morning_payout']:.0f} coins\n")
        f.write(f"Max Through Count (Morning -> Heaven): {results['max_morning_through']} consecutive initial hits\n")
        
        f.write("\n--- Continuation Rates (All Chains) ---\n")
        for stat in results['continuation_stats']:
            if stat['count_reached'] == 0: continue
            if stat['current'] > 10 and stat['count_reached'] < 5: continue
            
            f.write(f"{stat['current']} hits -> {stat['next']} hits: {stat['rate']:.1f}% ({stat['count_next']}/{stat['count_reached']})\n")
    print("Analysis complete. Results written to result.txt")

if __name__ == '__main__':
    data = load_data(FILE_PATH)
    machines = analyze_machines(data)
    results = calculate_metrics(machines)
    print_results(results)
