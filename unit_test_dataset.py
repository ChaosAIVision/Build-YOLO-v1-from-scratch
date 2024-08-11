from utils.dataset import  Create_YOLO_Cache, YOLODataset
yaml = '/home/chaos/Documents/ChaosAIVision/Build-YOLO-v1-from-scratch/data.yaml'
# Create an instance of Create_YOLO_Cache and save cache
# cache_creator = Create_YOLO_Cache(is_train='train', data_yaml=yaml)
# cache_creator.__save_cache__()

# import pickle
# import os

# # def load_cache(cache_file_path):
# #     """
# #     Load cache from a Pickle file.

# #     Args:
# #         cache_file_path (str): Path to the cache file.

# #     Returns:
# #         list: List of cached data loaded from the file.
# #     """
# #     if not os.path.isfile(cache_file_path):
# #         raise FileNotFoundError(f"The cache file {cache_file_path} does not exist.")
    
# #     with open(cache_file_path, 'rb') as f:
# #         cache = pickle.load(f)
    
# #     return cache

# # path = '/home/chaos/Documents/ChaosAIVision/dataset/fire_dataset/train/labels.cache'
# # a = load_cache(path)
# # print(type(a))
dataset= YOLODataset('train', yaml,7,2,20)
a = dataset.__getitem__(20)
print(a)