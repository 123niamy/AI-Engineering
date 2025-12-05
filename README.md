# Data Engineering Learning Projects

A collection of data engineering pipelines built while learning ETL concepts, data transformation, and workflow orchestration.

## 📚 Projects

### 1. Iris Dataset Pipeline (Complete 7-Stage ETL)
**File:** `Iris data pipeline1.py`

A comprehensive data engineering pipeline demonstrating the full ETL workflow:
- **Stage 1:** Data Ingestion (sklearn → pandas)
- **Stage 2:** Raw Storage (CSV)
- **Stage 3:** Preprocessing (missing values, scaling, encoding)
- **Stage 4:** Data Integration (metadata joins)
- **Stage 5:** Quality Validation (checks & reports)
- **Stage 6:** Governance & Security (metadata, masking)
- **Stage 7:** Data Serving (curated datasets, visualizations)

**Output:** Cleaned datasets, quality reports, correlation heatmaps, distribution plots

---

### 2. Titanic Survival Analysis Pipeline
**File:** `Data Engineering2.py`

End-to-end pipeline for the Titanic dataset with error handling and logging:
- Data ingestion with validation
- Missing value imputation (median/mode)
- Feature engineering (FamilySize)
- Categorical encoding (Sex, Embarked)
- SQLite storage
- SQL aggregations (survival rates by class/gender)
- Visualizations (survival rates, age distribution)

**Key Features:** Comprehensive error handling, logging for governance, file I/O safety

---

### 3. Simple CSV to SQLite Ingestion
**File:** `Data Engineering3.py`

Basic data ingestion pipeline:
- Load CSV with pandas
- Validate data schema
- Store in SQLite database
- Query and extract data

**Purpose:** Foundation for understanding data ingestion patterns

---

## 🛠️ Technologies Used

- **Python 3.14**
- **Pandas** - Data manipulation
- **SQLite3** - Database storage
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Dataset loading and preprocessing
- **Logging** - Pipeline monitoring and governance

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/data-engineering-learning.git
cd data-engineering-learning
```

2. Install dependencies:
```bash
pip install pandas matplotlib seaborn scikit-learn
```

3. Run a pipeline:
```bash
python "Iris data pipeline1.py"
python "Data Engineering2.py"
```

## 📊 Outputs

Each pipeline generates:
- Cleaned datasets (CSV/Parquet)
- SQLite databases
- Visualization plots (PNG)
- Quality reports (JSON)
- Execution logs

## 📖 Learning Objectives

- ✅ ETL pipeline design
- ✅ Data validation and quality checks
- ✅ Error handling best practices
- ✅ Data governance and logging
- ✅ Feature engineering
- ✅ Database operations
- ✅ Data visualization

## 🎯 Future Enhancements

- [ ] Add data profiling
- [ ] Implement data lineage tracking
- [ ] Add unit tests
- [ ] Cloud storage integration (AWS S3, Azure Blob)
- [ ] Real-time data ingestion
- [ ] API-based data sources

## 📝 License

MIT License - Feel free to use for learning purposes

---

**Status:** 🟢 Active Learning | Last Updated: December 2025
