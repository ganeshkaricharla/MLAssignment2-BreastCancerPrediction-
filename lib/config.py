import os

# Get the project root directory (parent of lib folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths relative to project root
TEST_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "test_data.csv")
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "wdbc.data")

# Model hyperparameters
LR_MAX_ITERATION = 2000
RANDOM_STATE = 42