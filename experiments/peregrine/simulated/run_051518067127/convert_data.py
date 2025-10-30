"""Convert JSON data to CSV format for analysis."""
import json
import pandas as pd
from pathlib import Path

# Load JSON data
with open('/workspace/data.json', 'r') as f:
    data = json.load(f)

# Create DataFrame
df = pd.DataFrame({
    'year': data['year'],
    'C': data['C']
})

# Save to CSV
df.to_csv('/workspace/data/data.csv', index=False)
print(f"Data converted: {len(df)} observations")
print(f"\nFirst few rows:")
print(df.head(10))
print(f"\nLast few rows:")
print(df.tail(10))
print(f"\nBasic statistics:")
print(df.describe())
