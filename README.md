# Data Organizer
A generic data-cleaning and normalization tool for **messy CSV / TXT / TSV files**.  
This script automatically detects encodings, separators, headers, and metadata blocks, then exports **cleaned CSV files and structured reports** for downstream analysis.

It is designed for engineering, research, and manufacturing environments where data files often contain:
- Mixed encodings (UTF-8, Shift-JIS, CP932, etc.)
- Irregular headers and metadata blocks
- Inconsistent separators (comma, tab, semicolon, pipe)
- Mixed numeric / datetime / string columns

---
## Overview
This tool scans a specified data folder, reads various text-based data files
(CSV, TXT, TSV), and automatically:

- Detects file encodings and separators
- Identifies table headers and skip rows
- Extracts metadata written before the data table
- Cleans and normalizes tabular data
- Converts datetime-like and numeric-like columns
- Outputs cleaned CSV files and summary reports

The goal is to eliminate manual preprocessing in Excel and provide
machine-readable, analysis-ready datasets.

---
## Included files
- **Data Organizer.py**  
  Main script that performs file scanning, parsing, cleaning, metadata extraction,
  and report generation.

---
## Typical use cases
- Cleaning equipment logs exported in inconsistent CSV/TXT formats
- Organizing experimental or measurement data with metadata headers
- Preparing data for:
  - Statistical analysis (SPC, DOE, FTA)
  - Visualization
  - Machine learning pipelines
- Batch-normalizing vendor-specific or legacy data files
- Auditing column structures and data types across many files

---
## Folder structure
project folder/
-├─ Data Organizer.py
-├─ DATA_FOLDER/ # Input data
-└─ OUTPUT/ # Auto-generated output


---
## Output
### output descriptions
- **cleaned_csv/**  
  Final, analysis-ready CSV files with normalized column names and cleaned values.
- **metadata/**  
  Metadata extracted from lines before the detected table header
  (e.g., conditions, notes, parameters).
- **report_summary.csv**  
  Overview of parsing status, encoding, separators, row/column counts.
- **columns_inventory.csv**  
  Useful for understanding schema differences across multiple data files.

---
## Limitations and notes
- Datetime and numeric conversion are heuristic-based, not guaranteed
- Files with no clear tabular structure may be skipped or partially parsed
- This tool focuses on preprocessing, not validation of data correctness
