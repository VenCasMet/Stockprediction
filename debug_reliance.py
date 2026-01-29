from main import run_full_analysis
import sys

# Redirect stdout to avoid clutter, will print checks at end
try:
    print("Running analysis for RELIANCE.NS...")
    # NOTE: Passing just ticker string as run_full_analysis signature in main.py is (ticker, fast_mode)
    results = run_full_analysis("RELIANCE.NS", fast_mode=True)
    
    if results.get("error"):
        print(f"Error: {results.get('message')}")
    else:
        print(f"\n--- LOCAL MODEL RESULTS (Reliance) ---")
        print(f"Decision: {results['decision']}")
        
        # Verify Chart Data
        chart_data = results.get("chart_data")
        
        with open("chart_debug_report.txt", "w") as f:
            if chart_data:
                f.write(f"[Chart Data Check]\n")
                dates = chart_data.get("dates")
                prices = chart_data.get("actual_prices")
                
                f.write(f"Dates Count: {len(dates) if dates else 'None'}\n")
                f.write(f"Prices Count: {len(prices) if prices else 'None'}\n")
                
                if dates and len(dates) > 0:
                    f.write(f"Sample Date[0]: '{dates[0]}' (Type: {type(dates[0])})\n")
                    f.write(f"Sample Date[-1]: '{dates[-1]}'\n")
                
                if not dates or len(dates) == 0:
                     f.write("!!! ERROR: Chart dates are empty!\n")
                else:
                     f.write("Chart data looks valid.\n")
                     
                # Check models inside chart_data
                if "models" in chart_data:
                    f.write(f"Models in chart: {list(chart_data['models'].keys())}\n")
                    if "ensemble" in chart_data["models"]:
                         ens = chart_data["models"]["ensemble"]
                         f.write(f"Ensemble Forecast: {ens.get('forecast_7day')}\n")
            else:
                f.write("!!! ERROR: chart_data is None!\n")
                
        print("Debug report written to chart_debug_report.txt")
        
except Exception as e:
    print(f"Script Error: {e}")
    import traceback
    traceback.print_exc()
