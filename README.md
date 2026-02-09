# KPI Delivery Performance (Pandas)

## What it does
- Merge X/y per ID
- Data-quality checks (duplicates, missingness)
- Segment KPIs + bucketing (discount/weight)
- Impact prioritization (gap × volume)

## Setup
pip install -r requirements.txt

python src/kpi_analysis.py

## Data
Files are included in `data/`:
- X_train.csv
- y_train.csv

## Run
python src/kpi_analysis.py

## Output

CSV results are saved to `outputs/`.


