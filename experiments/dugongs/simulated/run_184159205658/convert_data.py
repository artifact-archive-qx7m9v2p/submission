import json
import pandas as pd
from pathlib import Path

# Load JSON data
with open('/workspace/data.json', 'r') as f:
    data_dict = json.load(f)

# Convert to DataFrame
df = pd.DataFrame({'x': data_dict['x'], 'Y': data_dict['Y']})

# Save to CSV
df.to_csv('/workspace/data/data.csv', index=False)

print(f"Data converted: {len(df)} observations")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nBasic statistics:")
print(df.describe())
