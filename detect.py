import pickle
import os
def load_cache(cache_file_path):
    """
    Load cache from a Pickle file.

    Args:
        cache_file_path (str): Path to the cache file.

    Returns:
        list: List of cached data loaded from the file.
    """
    if not os.path.isfile(cache_file_path):
        raise FileNotFoundError(f"The cache file {cache_file_path} does not exist.")
    
    with open(cache_file_path, 'rb') as f:
        cache = pickle.load(f)
    
    return cache
path = '/home/chaos/Documents/ChaosAIVision/dataset/voc_dataset/train/labels.cache'
a = load_cache(path)
print(a)