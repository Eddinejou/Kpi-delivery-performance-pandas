import pandas as pd
import os

# Project root folder (folder where this .py file is located :
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Output folder inside the project (auto-created)
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

#Reading the csv files
# Put X_train.csv and y_train.csv inside a folder named "data" in the same project
X = pd.read_csv(os.path.join(BASE_DIR, "data", "X_train.csv"))
Y = pd.read_csv(os.path.join(BASE_DIR, "data", "y_train.csv"))
#############
#print("X shape:", X.shape)
#print("Y shape:", Y.shape)

#print("X columns:", list(X.columns)[:30]) # first 30 columns
#print("Y columns:", list(Y.columns))

#print("\nX head:")
#print(X.head())
#print("\nY head:")
#print(Y.head())

# check ID uniqueness :
#print("Y unique IDs:", Y['ID'].nunique(), "rows:", len(Y))
#print("X unique IDs:", X['ID'].nunique(), "rows:", len(X))

# IDs present in Y but not in X and vice versa :
#ids_y_not_x = set(Y['ID']) - set(X['ID'])
#ids_x_not_y = set(X['ID']) - set(Y['ID'])
#print("IDs in Y not in X (sample up to 10):", list(ids_y_not_x)[:10])
#print("IDs in X not in Y (sample up to 10):", list(ids_x_not_y)[:10])

# merge :
df = X.merge(Y, on='ID', how='inner')
#print(df)
#print("Merged shape (inner):", df.shape)

# backup and quick snapshot
df_orig = df.copy()
print("Rows, columns:", df.shape)
print("First 3 rows:")
print(df.head(3))
                              #### Cleaning Data  ####
# Remove exact duplicate rows

before = len(df)
df = df.drop_duplicates()
after = len(df)
print("Dropped exact duplicate rows:", before - after)
print("New shape:", df.shape)

if before == after:
    print("no duplicate rows found")

# Check duplicate IDs :

dup_id_count = df['ID'].duplicated().sum()
print("Rows with duplicated ID:", dup_id_count)
if dup_id_count > 0:
    dup_groups = df[df.duplicated(subset=['ID'], keep=False)].sort_values('ID').head(20)
    print("\nSample duplicate ID rows:")
    print(dup_groups)
else:
    print("No duplicate IDs found.")

# show missingness and drop columns with >60% missing :

missing_frac = df.isna().mean().sort_values(ascending=False)
print("Top columns by missing fraction:\n", missing_frac.head(20))
to_drop = missing_frac[missing_frac > 0.6].index.tolist()
print("\nColumns to drop (>60% missing):", to_drop)
df = df.drop(columns=to_drop)
print("\nNew shape after dropping high-missing columns:", df.shape)

                              ####  KPI  ####
pd.options.display.float_format = '{:.3f}'.format
# Overall :
overall_rate = df['Reached.on.Time_Y.N'].mean()
counts = df['Reached.on.Time_Y.N'].value_counts().sort_index()
print("Overall on-time rate:", overall_rate)
print("\nLabel counts (0 = late, 1 = on-time):\n", counts)

# By Mode_of_Shipment :warehouse_block, Product_importance, Gender

for col in ['Mode_of_Shipment', 'Warehouse_block', 'Product_importance', 'Gender']:
    if col in df.columns:
        print(f"\nOn-time rate by {col}:")
        print(df.groupby(col)['Reached.on.Time_Y.N'].agg(n_orders='count', on_time_rate='mean').sort_values('on_time_rate', ascending=False))

# discount buckets
df['discount_bin'] = pd.cut(df['Discount_offered'], bins=[-0.01,0,5,10,20,100], labels=['0%','0-5%','5-10%','10-20%','20%+'])
print(df.groupby('discount_bin')['Reached.on.Time_Y.N'].agg(n_orders='count', on_time_rate='mean').sort_values('on_time_rate'))

# weight buckets
if 'Weight_in_gms' in df.columns:
    try:
        df['weight_bin_q'] = pd.qcut(df['Weight_in_gms'], q=4, labels=['Q1','Q2','Q3','Q4'])
    except (ValueError, IndexError):
        df['weight_bin_q'] = pd.cut(df['Weight_in_gms'], bins=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    # show counts so you can confirm the column was created
print("weight_bin_q value counts:\n", df['weight_bin_q'].value_counts(dropna=False))

       #### small check ####

#### how many orders in 20%+ bucket and a sample :

#k = df[df['discount_bin'] == '20%+']
#print("Count in 20%+ bucket:", len(k))
#print(k[['ID','Discount_offered','Reached.on.Time_Y.N','Warehouse_block','Product_importance','Mode_of_Shipment']].head(20))

#### Checking the relevancy of the discount in correlation to the ['Reached.on.Time_Y.N'] :

#k = df[df['discount_bin'] == '20%+']
#print("Count in 20%+ bucket:", len(k))
#print("Counts by target in 20%+ bucket:")
#print(k['Reached.on.Time_Y.N'].value_counts(dropna=False))

# Gap and impact for two key sectors: Warehouse_block and Mode_of_Shipment
for col in ['Reached.on.Time_Y.N','Weight_in_gms','Customer_rating','Prior_purchases','Discount_offered']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
# Overall on-time rate
overall = df['Reached.on.Time_Y.N'].mean()
# Generic impact table function
def impact_table(df, col, low_conf_threshold=50):
    if col not in df.columns:
        return pd.DataFrame(columns=[col, 'n_orders','on_time_rate','gap','impact','low_confidence'])
    summary = df.groupby(col)['Reached.on.Time_Y.N'].agg(n_orders='count', on_time_rate='mean').reset_index()
    summary['gap'] = overall - summary['on_time_rate']       # percentage-point gap
    summary['impact'] = summary['gap'] * summary['n_orders'] # gap * volume
    summary['low_confidence'] = summary['n_orders'] < low_conf_threshold
    return summary.sort_values('impact', ascending=False).reset_index(drop=True)
# Compute tables
warehouse_impact = impact_table(df, 'Warehouse_block')
mode_impact = impact_table(df, 'Mode_of_Shipment')
# Display results
print("Overall on-time rate: {:.3f}\n".format(overall))
print("### Warehouse_block impact (sorted by impact) ===")
print(warehouse_impact.to_string(index=False))
print("\n### Mode_of_Shipment impact (sorted by impact) ===")
print(mode_impact.to_string(index=False))

# discount_kpi
if 'discount_bin' in df.columns:
    discount_kpi = df.groupby('discount_bin')['Reached.on.Time_Y.N'] \
    .agg(n_orders='count', on_time_rate='mean').reset_index().sort_values('discount_bin') # weight_kpi if 'weight_bin_q' in df.columns: weight_kpi = df.groupby('weight_bin_q')['Reached.on.Time_Y.N'] \ .agg(n_orders='count', on_time_rate='mean').reset_index().sort_values('weight_bin_q')

# weight_kpi
if 'weight_bin_q' in df.columns:
    weight_kpi = df.groupby('weight_bin_q')['Reached.on.Time_Y.N'] \
    .agg(n_orders='count', on_time_rate='mean').reset_index().sort_values('weight_bin_q')


# Save individual CSVs

if 'warehouse_impact' in globals() and not warehouse_impact.empty:
    warehouse_impact.to_csv(os.path.join(OUT_DIR, "warehouse_impact.csv"), index=False)
    print("Saved warehouse_impact.csv")

if 'mode_impact' in globals() and not mode_impact.empty:
    mode_impact.to_csv(os.path.join(OUT_DIR, "mode_impact.csv"), index=False)
    print("Saved mode_impact.csv")

if 'discount_kpi' in globals() and not discount_kpi.empty:
    discount_kpi.to_csv(os.path.join(OUT_DIR, "discount_kpi.csv"), index=False)
    print("Saved discount_kpi.csv")

if 'weight_kpi' in globals() and not weight_kpi.empty:
    weight_kpi.to_csv(os.path.join(OUT_DIR, "weight_kpi.csv"), index=False)
    print("Saved weight_kpi.csv")
