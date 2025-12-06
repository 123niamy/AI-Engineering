# Data Engineering workflow:
# 1. Data Ingestion
import pandas as pd
import sqlite3
import sys
import argparse
from pathlib import Path

def ingest_csv_to_sqlite(csv_path, db_path, table_name):
    """
    Ingest CSV data into SQLite database with validation and error handling.
    
    Args:
        csv_path: Path to the CSV file
        db_path: Path to the SQLite database
        table_name: Name of the table to create/replace
    
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        # Step 1: Validate file exists
        if not Path(csv_path).exists():
            print(f"Error: File '{csv_path}' not found")
            return None
        
        # Step 2: Read CSV
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} rows from {csv_path}")
        
        # Step 3: Validate (basic checks)
        if df.empty:
            print("Warning: DataFrame is empty")
            return None
        
        print(f"✓ Columns ({len(df.columns)}): {df.columns.tolist()}")
        
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            print("⚠ Missing values per column:")
            print(missing_counts[missing_counts > 0])
        else:
            print("✓ No missing values detected")
        
        # Step 4: Persist into SQLite using context manager
        with sqlite3.connect(db_path) as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        print(f"✓ Data ingested into '{table_name}' in {db_path}")
        print(f"✓ Data types: {df.dtypes.to_dict()}")
        
        return df
    
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file '{csv_path}' is empty")
        return None
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse CSV - {e}")
        return None
    except sqlite3.Error as e:
        print(f"Error: Database error - {e}")
        return None
    except Exception as e:
        print(f"Error: Unexpected error - {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Ingest CSV data into SQLite database")
    parser.add_argument("csv_path", nargs="?", default="iris.csv", help="Path to CSV file")
    parser.add_argument("--db", default="pipeline.db", help="Path to SQLite database")
    parser.add_argument("--table", default="iris_data", help="Table name in database")
    
    args = parser.parse_args()
    
    print(f"Starting data ingestion...")
    print(f"CSV: {args.csv_path}")
    print(f"Database: {args.db}")
    print(f"Table: {args.table}")
    print("-" * 50)
    
    df = ingest_csv_to_sqlite(args.csv_path, args.db, args.table)
    
    if df is not None:
        print("-" * 50)
        print("✓ Ingestion completed successfully!")
        print(f"Shape: {df.shape}")
    else:
        print("-" * 50)
        print("✗ Ingestion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
