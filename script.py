import time
import fireducks as fd
from fireducks.core import get_fireducks_options
import fireducks.pandas as fpd
import pandas as pd
import os

DATA_FILE = "data.csv"

# Ensure 'data.csv' exists or create a fresh one
if os.path.exists(DATA_FILE):
    os.remove(DATA_FILE)  # Delete old CSV to ensure fresh creation

print(f"Creating a new sample '{DATA_FILE}' with target_column.", flush=True)
data = {
    "column_a": [1, 2, 3, 4, 5],
    "column_b": [10, 20, 30, 40, 50],
    "target_column": [100, 200, 300, 400, 500]  # Ensure target_column exists
}
pd.DataFrame(data).to_csv(DATA_FILE, index=False)
print(f"Sample '{DATA_FILE}' created successfully!", flush=True)

def main():
    print("--- FireDucks Lazy Execution Demo ---", flush=True)

    # Lazy execution example
    t0 = time.time()
    df = fpd.read_csv(DATA_FILE)._evaluate()  # Ensure fresh read
    t1 = time.time()
    df = df.sort_values("column_a")._evaluate()
    t2 = time.time()
    df.to_csv("sorted.csv") 
    t3 = time.time()

    print("Time to read CSV:", t1 - t0, flush=True)
    print("Time to sort values:", t2 - t1, flush=True)
    print("Time to save to CSV:", t3 - t2, flush=True)

    # Enable Benchmark Mode
    get_fireducks_options().set_benchmark_mode(True)
    print("Benchmark mode enabled.", flush=True)

    # Read data again to ensure it's updated
    fd_df = fpd.read_csv(DATA_FILE)._evaluate()
    print("Updated FireDucks DataFrame Columns:", fd_df.columns, flush=True)

    # Error handling for missing columns
    if "target_column" not in fd_df.columns:
        print("Error: 'target_column' still missing! Check CSV formatting.", flush=True)
        return  # Exit the function to prevent further errors

    # FIX: Use Pandas for Aggregation
    try:
        print("Converting FireDucks DataFrame to Pandas for aggregation...", flush=True)
        pd_df = fd_df.to_pandas()  # Convert to Pandas

        # Perform aggregation with Pandas
        aggregated = pd_df.groupby("column_a")["target_column"].agg(["sum", "mean"]).reset_index()

        print("Aggregation successful! Here is the result:\n", aggregated, flush=True) 

        # Convert back to FireDucks DataFrame if needed
        fd_aggregated = fpd.from_pandas(aggregated)._evaluate()
    except Exception as e:
        print("Aggregation failed! Error:", e, flush=True)

    # Multi-Target Encoding with FireDucks
    try:
        encoded = fpd.feat.multi_target_encoding(fd_df, ['column_b'], 'target_column', 'mean')
        print("Multi-target encoding successful.", flush=True)
    except TypeError as e:
        print(f"multi_target_encoding error: {e}", flush=True)

    print("Feature engineering operations completed.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Script crashed with error:", e, flush=True)


