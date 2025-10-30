import numpy as np

def convert_to_python_types(obj):
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Test
coverage_results = {
    0.5: {'expected': 0.5, 'observed': 0.519, 'n_in': np.int64(14), 'n_total': 27}
}
print(convert_to_python_types(coverage_results))
