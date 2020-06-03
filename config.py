import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
Data_path = os.path.join(BASE_PATH, r"data/")
App_path = os.path.join(BASE_PATH, r"data/app")
processed_path = os.path.join(BASE_PATH, r"preprocessing/processed_files")
num_classes = 12
operators = ['+', '-']
labels = [10, 11]
