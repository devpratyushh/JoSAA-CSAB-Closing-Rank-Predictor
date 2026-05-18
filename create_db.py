import sqlite3
import pandas as pd
import os

def build_db():
    db_path = "historical.db"
    
    # Connect to SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    
    # Load CSAB
    if os.path.exists("csab_ranks.csv"):
        print("Loading csab_ranks.csv...")
        df_csab = pd.read_csv("csab_ranks.csv", dtype=str)
        # The CSV columns match the expected schema
        df_csab.to_sql("csab", conn, if_exists="replace", index=False)
        print(f"Loaded {len(df_csab)} rows into 'csab' table.")
    
    # Load JoSAA
    if os.path.exists("josaa_ranks.csv"):
        print("Loading josaa_ranks.csv...")
        df_josaa = pd.read_csv("josaa_ranks.csv", dtype=str)
        df_josaa.to_sql("josaa", conn, if_exists="replace", index=False)
        print(f"Loaded {len(df_josaa)} rows into 'josaa' table.")

    conn.close()
    print("Database built successfully!")

if __name__ == "__main__":
    build_db()
