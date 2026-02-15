import pandas as pd
from sklearn.model_selection import train_test_split
from constants import DROP_FIELDS, TARGET_FIELD,COLUMN_NAMES
from config import TEST_FILE_PATH, DATASET_PATH

def get_column_names():
    column_names = COLUMN_NAMES
    return column_names

def load_dataset(filepath=DATASET_PATH):
    columns = get_column_names()
    df = pd.read_csv(filepath,header=None, names=columns)
    return df

def load_data_and_save_test(filepath=DATASET_PATH, testsize=0.2,randomstate=42):
    df = load_dataset(filepath)
    X = df.drop(DROP_FIELDS, axis=1)
    y = df[TARGET_FIELD]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testsize, random_state=randomstate, stratify=y
    )

    test_data = X_test.copy()
    test_data[TARGET_FIELD] = y_test
    test_data.to_csv(TEST_FILE_PATH, index=False)

    return X_train, X_test, y_train, y_test