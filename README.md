# ML Assignment 2

## Breast Cancer Wisconsin - CLASSIFICATION

## Problem statement.

The objective of the project is to build and evaluate multiple ml models to predict whether a breast tumer is Malignent or Benign, based on the features extracted from digitized images of breast masses.
Six classification models are implemented & evaluted

- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Random Forest
- XGBoost

## Dataset Description

Data set name: Breast Cancer winsconsin
Source: UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Instances: 569
Features: 30
Target variable: diagnosis
Missing values: None

## Models Used

### Comparison Table

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression      | 0.9649   | 0.9960 | 0.9750    | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree            | 0.9298   | 0.9246 | 0.9048    | 0.9048 | 0.9048 | 0.8492 |
| KNN                      | 0.9561   | 0.9823 | 0.9744    | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes (Gaussian)   | 0.9211   | 0.9891 | 0.9231    | 0.8571 | 0.8889 | 0.8292 |
| Random Forest (Ensemble) | 0.9737   | 0.9929 | 1.0000    | 0.9286 | 0.9630 | 0.9442 |
| XGBoost (Ensemble)       | 0.9649   | 0.9967 | 1.0000    | 0.9048 | 0.9500 | 0.9258 |

### Observations

| ML Model Name            | Observation about model performance                                                                                                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Works really well with 96.49% accuracy after feature scaling. Good balance of precision (0.97) and recall (0.93), and a very high AUC of 0.996. Scaling was needed for the optimizer to converge properly.                |
| Decision Tree            | Weakest performer with 92.98% accuracy and lowest AUC of 0.92. It overfits the training data, which limits generalization. MCC of 0.85 is the lowest among all models.                                                    |
| KNN                      | Solid 95.61% accuracy after scaling. High precision of 0.97 means very few false positives, though recall of 0.90 means some malignant cases are missed. Scaling helped a lot since it uses distances across 30 features. |
| Naive Bayes (Gaussian)   | Good at ranking probabilities with AUC of 0.989, but has the lowest recall at 0.86 — it misses more malignant cases than other models. Overall accuracy is 92.11%.                                                        |
| Random Forest (Ensemble) | Best model overall — highest accuracy at 97.37%, perfect precision (1.0), and best MCC of 0.94. Combining 100 trees reduces overfitting and gives the most reliable predictions.                                          |
| XGBoost (Ensemble)       | Very close to Random Forest with 96.49% accuracy and the best AUC of 0.997. Also has perfect precision (1.0). Its boosting approach gives the best probability estimates among all models.                                |
