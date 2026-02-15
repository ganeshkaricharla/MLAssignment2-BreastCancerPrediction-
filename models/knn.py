# Decision Tree

# Logistic Regression

import sys
import os

# Add lib directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

from sklearn.neighbors import KNeighborsClassifier
import joblib
from data_utils import load_data_and_save_test
from metrics import calculate_all_metrics, display_metrics, get_confusion_matrix, get_classification_report
from config import KNN_N_NEIGHBORS, KNN_WEIGHTS, KNN_METRIC
from constants import POS_LABEL

class KNNModel():

    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = load_data_and_save_test()
        self.model = KNeighborsClassifier(
            n_neighbors=KNN_N_NEIGHBORS, 
            weights=KNN_WEIGHTS,
            metric=KNN_METRIC
            )

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self):
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:,1]
        return y_pred, y_pred_proba
    
    def metrics(self, y_pred, y_pred_proba):
        self.metrics = calculate_all_metrics(self.y_test, y_pred,y_pred_proba, pos_label=POS_LABEL)
        display_metrics(self.metrics)
        return self.metrics
        
    def get_confusion_metrics(self,y_pred):
        self.cm = get_confusion_matrix(self.y_test, y_pred)
        return self.cm
    
    def get_classification_report(self, y_pred):
        self.classification_report = get_classification_report(self.y_test, y_pred)
        return self.classification_report
    
    def save(self, model_file):
        joblib.dump(self.model, model_file)

if __name__ == "__main__":
    dt = KNNModel()
    dt.train()
    y_pred, y_pred_proba = dt.predict()
    metrics = dt.metrics(y_pred=y_pred, y_pred_proba=y_pred_proba)
    confusion_matrix = dt.get_confusion_metrics(y_pred=y_pred)
    classification_report = dt.get_classification_report(y_pred=y_pred)

