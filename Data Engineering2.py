import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR)  # Change to script directory

# -------------------------------
# 1. Governance: Logging setup
# -------------------------------
logging.basicConfig(
    filename="titanic_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------
# 2. Ingestion
# -------------------------------
def ingest(csv_path):
    try:
        from pathlib import Path
        if not Path(csv_path).exists():
            logging.error(f"File not found: {csv_path}")
            raise FileNotFoundError(f"CSV file '{csv_path}' does not exist")
        
        logging.info("Ingesting data from CSV")
        df = pd.read_csv(csv_path)
        
        if df.empty:
            logging.warning("CSV file is empty")
            raise ValueError("CSV file contains no data")
        
        return df
    except Exception as e:
        logging.error(f"Ingestion failed: {e}")
        raise

# -------------------------------
# 3. Validation
# -------------------------------
def validate(df):
    try:
        logging.info("Validating dataset")
        required_cols = ['Age', 'Sex', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Embarked']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            logging.error(f"Missing required columns: {missing}")
            raise ValueError(f"Missing columns: {missing}")
        
        logging.info("Validation passed")
        return df
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        raise

# -------------------------------
# 4. Transformation
# -------------------------------
def transform(df):
    try:
        logging.info("Transforming dataset")
        df = df.copy()
        
        # Handle missing values
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna('S')
        
        # Feature engineering
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Encoding
        df['Sex'] = df['Sex'].map({'male':0, 'female':1})
        df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})
        
        logging.info(f"Transformation complete. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Transformation failed: {e}")
        raise

# -------------------------------
# 5. Storage
# -------------------------------
def store(df, db_path="titanic.db"):
    try:
        logging.info("Storing cleaned dataset into SQLite")
        with sqlite3.connect(db_path) as conn:
            df.to_sql("titanic_cleaned", conn, if_exists="replace", index=False)
        logging.info(f"Data stored successfully in {db_path}")
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        raise
    except Exception as e:
        logging.error(f"Storage failed: {e}")
        raise

# -------------------------------
# 6. Processing
# -------------------------------
def process(db_path="titanic.db"):
    try:
        logging.info("Processing data with SQL queries")
        with sqlite3.connect(db_path) as conn:
            query = """
            SELECT Sex, Pclass, AVG(Survived) as survival_rate
            FROM titanic_cleaned
            GROUP BY Sex, Pclass
            """
            result = pd.read_sql(query, conn)
        logging.info("Processing complete")
        return result
    except sqlite3.Error as e:
        logging.error(f"Database query error: {e}")
        raise
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        raise

# -------------------------------
# 7. Visualization
# -------------------------------
def visualize(df):
    try:
        logging.info("Visualizing survival rates")
        
        # Plot 1: Survival by class and gender
        plt.figure(figsize=(8,6))
        sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df)
        plt.title("Survival Rate by Class and Gender")
        plt.savefig("survival_by_class_gender.png")
        plt.close()
        logging.info("Saved: survival_by_class_gender.png")

        # Plot 2: Age distribution
        plt.figure(figsize=(8,6))
        sns.histplot(df['Age'], bins=20, kde=True)
        plt.title("Age Distribution of Passengers")
        plt.savefig("age_distribution.png")
        plt.close()
        logging.info("Saved: age_distribution.png")
        
    except Exception as e:
        logging.error(f"Visualization failed: {e}")
        raise

# -------------------------------
# Run the pipeline
# -------------------------------
if __name__ == "__main__":
    try:
        print("Starting Titanic Data Pipeline...")
        logging.info("Pipeline started")
        
        raw_df = ingest("titanic.csv")
        print(f"✓ Ingested {len(raw_df)} rows")
        
        validated_df = validate(raw_df)
        print("✓ Validation passed")
        
        transformed_df = transform(validated_df)
        print("✓ Transformation complete")
        
        store(transformed_df)
        print("✓ Data stored in database")
        
        summary = process()
        print("\n✓ Processing complete. Survival rates:")
        print(summary)
        
        visualize(transformed_df)
        print("✓ Visualizations saved")
        
        logging.info("Pipeline completed successfully")
        print("\n✅ Pipeline completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure 'titanic.csv' exists in the current directory.")
        logging.error(f"Pipeline failed: {e}")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logging.error(f"Pipeline failed with exception: {e}", exc_info=True)
