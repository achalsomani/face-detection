from pathlib import Path

PROJECT_PATH = Path(__file__).parent
DATA_PATH = PROJECT_PATH / "trainingset"
CSV_FILE = DATA_PATH / "train_data_mapping.csv"
TESTING_DATA_PATH = PROJECT_PATH / "testing_data"
TESTING_CSV_FILE = TESTING_DATA_PATH / "test_data_mapping.csv"