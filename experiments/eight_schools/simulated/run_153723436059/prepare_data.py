"""Convert JSON data to CSV format for analysis."""
import json
import pandas as pd
from pathlib import Path

# Load JSON data
data_path = Path("/workspace/data.json")
with open(data_path) as f:
    raw_data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame({
    'school': range(1, raw_data['J'] + 1),
    'effect': raw_data['y'],
    'sigma': raw_data['sigma']
})

# Save to data directory
output_path = Path("/workspace/data/data.csv")
df.to_csv(output_path, index=False)

print(f"Data converted and saved to {output_path}")
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nSummary statistics:")
print(df.describe())
