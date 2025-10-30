"""Convert JSON data to CSV format for analysis."""
import json
import pandas as pd
from pathlib import Path

# Read JSON data
with open('/workspace/data.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame({
    'group': range(1, data['N'] + 1),
    'n': data['n'],
    'r': data['r']
})

# Add derived columns for analysis
df['proportion'] = df['r'] / df['n']
df['failures'] = df['n'] - df['r']

# Save to CSV
output_path = Path('/workspace/data/data.csv')
df.to_csv(output_path, index=False)

print(f"Data converted and saved to {output_path}")
print(f"\nDataFrame shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData summary:")
print(df.describe())
