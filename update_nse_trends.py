import sys
import os
import json
import concurrent.futures
from datetime import datetime

# Import necessary functions from main.py
try:
    from main import predict_single_stock, STATIC_DIR
except ImportError:
    # Fix path if needed
    sys.path.append(os.getcwd())
    from main import predict_single_stock, STATIC_DIR

# Symbols extracted from NSE "Most Active" page via Browser Agent
# Appending .NS for Yahoo Finance compatibility
NSE_SYMBOLS = [
    'KAYNES.NS', 'INDIGO.NS', 'DIXON.NS', 'BSE.NS', 'SHAKTIPUMP.NS', 
    'IDEA.NS', 'HDFCBANK.NS', 'HINDZINC.NS', 'KOTAKBANK.NS', 
    'DCMSHRIRAM.NS', 'AEQUS.NS', 'ICICIBANK.NS', 'INFY.NS', 
    'RELIANCE.NS', 'TRENT.NS', 'BHARTIARTL.NS', 'SILVERBEES.NS', 'SWIGGY.NS'
]

def update_with_nse_data():
    print(f"Starting NSE Analysis for {len(NSE_SYMBOLS)} stocks...")
    results = []
    
    # Run predictions SEQUENTIALLY to debug overlap
    for sym in NSE_SYMBOLS[:8]: # Test first 8
        try:
            res = predict_single_stock(sym)
            if res and res['score'] > -999: 
                results.append(res)
                print(f"Processed: {sym} -> Price: {res['price']} | Score: {res['score']:.2f}%")
        except Exception as e:
            print(f"Error processing {sym}: {e}")

    # Rank by potential upside (score)
    results.sort(key=lambda x: x['score'], reverse=True)
    top_picks = results[:7] # Keep Top 7
    
    output_data = {
        "generated_at": datetime.now().isoformat(),
        "source": "NSE Most Active (Manual Update)",
        "stocks": top_picks
    }
    
    json_path = os.path.join(STATIC_DIR, "trending_stocks.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\nSuccess! Updated trending_stocks.json with {len(top_picks)} NSE stocks.")
    print("Top Picks with Outlook:")
    for stock in top_picks:
        print(f"- {stock['symbol']}: {stock.get('outlook', 'N/A')} | Exp Move: {stock.get('expected_move', 0)}")

if __name__ == "__main__":
    update_with_nse_data()
