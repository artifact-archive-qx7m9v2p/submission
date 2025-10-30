"""Convert JSON data to CSV format for analysis."""

import json
import pandas as pd
from pathlib import Path

# Load JSON data
with open('data.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame({
    'year': data['year'],
    'C': data['C']
})

# Save to data directory
df.to_csv('data/data.csv', index=False)

print(f"Data converted successfully!")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nBasic statistics:")
print(df.describe())
