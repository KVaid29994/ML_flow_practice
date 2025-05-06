import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt 
import seaborn as sns
import datetime
import pandas as pd
import sklearn



wine = load_wine()

X = wine.data
y = wine.target


X_train, X_test,y_train,y_test = train_test_split(X,y, test_size= 0.10, random_state = 42)

param_grid = {
    'max_depth': [5, 10, 15],
    'n_estimators': [10, 20, 50]
}

with mlflow.start_run():

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    #log best parameters
    for param_name, param_value in grid_search.best_params_.items():
        mlflow.log_param(param_name, param_value)

    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_param("cv_folds", 5)


    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_weighted", report['weighted avg']['precision'])
    mlflow.log_metric("recall_weighted", report['weighted avg']['recall'])
    mlflow.log_metric("f1_weighted", report['weighted avg']['f1-score'])

    cv_scores = cross_val_score(best_model,X_train,y_train, cv =5)
    mlflow.log_metric("cv_mean_accuracy", np.mean(cv_scores))
    mlflow.log_metric("cv_std_accuracy", np.std(cv_scores))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

     # save plot
    plt.savefig("Confusion-matrix.png")
    input_example = pd.DataFrame(X_test[:5], columns=wine.feature_names)
    # log artifacts using mlflow
    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({
        "Author": "Kashish",
        "Project": "Wine Classification",
        "Experiment_Date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "Scikit_Learn_Version": sklearn.__version__
    })

    # Log the model
    mlflow.sklearn.log_model(best_model, "Random-Forest-Model", input_example= input_example)

    print(accuracy)