"""Prepare data for analysis - convert JSON to CSV and create copies for parallel analysts."""
import json
import pandas as pd
from pathlib import Path

# Load JSON data
with open('data/data.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame({
    'year': data['year'],
    'C': data['C']
})

# Save main CSV
df.to_csv('data/data.csv', index=False)
print(f"Saved main data: {len(df)} observations")
print(f"Variables: {list(df.columns)}")
print(f"\nData preview:")
print(df.head())
print(f"\nData summary:")
print(df.describe())

# Create copies for 3 parallel EDA analysts
for i in range(1, 4):
    df.to_csv(f'data/data_analyst_{i}.csv', index=False)
    print(f"Created copy: data/data_analyst_{i}.csv")
