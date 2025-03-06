import numpy as np

def convert_to_serializable(obj):
    """Convert numpy types and ignore non-serializable objects."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):  # Convert numpy int
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):  # Convert numpy float
        return float(obj)
    elif isinstance(obj, np.ndarray):  # Convert numpy array
        return obj.tolist()
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple, set)):  # Convert lists/sets/tuples recursively
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):  # Convert dict recursively
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return None  # Ignore non-serializable objects