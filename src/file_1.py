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
from sklearn.neural_network import MLPClassifier
import sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")
wine = load_wine()

X = wine.data
y = wine.target


X_train, X_test,y_train,y_test = train_test_split(X,y, test_size= 0.10, random_state = 42)



param_grid_rf= {
    'max_depth': [5, 10, 15,20],
    'n_estimators': [10, 20, 30, 50]
}
param_grid_nn = {   'hidden_layer_sizes': [(50,), (100,), (50,50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam']
}

mlflow.set_experiment('MLops_experiment_1')


with mlflow.start_run():
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("train_fraction", len(X_train) / (len(X_train) + len(X_test)))
    mlflow.log_param("test_fraction", len(X_test) / (len(X_train) + len(X_test)))
    grid_search_rf= GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)

    best_rf = grid_search_rf.best_estimator_

    y_pred_rf = best_rf.predict(X_test)
    report = classification_report(y_test, y_pred_rf, output_dict=True)

    #log best parameters
    for param_name, param_value in grid_search_rf.best_params_.items():
        mlflow.log_param(param_name, param_value)

    accuracy = accuracy_score(y_test, y_pred_rf)
    mlflow.log_param("cv_folds", 5)


    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_weighted", report['weighted avg']['precision'])
    mlflow.log_metric("recall_weighted", report['weighted avg']['recall'])
    mlflow.log_metric("f1_weighted", report['weighted avg']['f1-score'])

    cv_scores = cross_val_score(best_rf,X_train,y_train, cv =5)
    mlflow.log_metric("cv_mean_accuracy", np.mean(cv_scores))
    mlflow.log_metric("cv_std_accuracy", np.std(cv_scores))

    ## creating a confusion metric log
    cm = confusion_matrix(y_test, y_pred_rf)
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
    mlflow.sklearn.log_model(best_rf, "Random-Forest-Model", input_example= input_example)

    print(accuracy)

mlflow.end_run()

with mlflow.start_run(run_name= "Neural Network"):
    grid_search_nn = GridSearchCV(MLPClassifier(random_state=42, max_iter=500), param_grid_nn,cv=5, scoring = 'accuracy', n_jobs= -1)
    grid_search_nn.fit(X_train, y_train)
    best_nn = grid_search_nn.best_estimator_

    y_pred_nn = best_nn.predict(X_test)
    report_nn = classification_report(y_test, y_pred_nn, output_dict=True)

    for param_name, param_value in grid_search_nn.best_params_.items():
        mlflow.log_param(f"nn_{param_name}", param_value)

    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    mlflow.log_metric("nn_accuracy", accuracy_nn)
    mlflow.log_metric("nn_precision_weighted", report_nn['weighted avg']['precision'])
    mlflow.log_metric("nn_recall_weighted", report_nn['weighted avg']['recall'])
    mlflow.log_metric("nn_f1_weighted", report_nn['weighted avg']['f1-score'])

    cv_scores_nn = cross_val_score(best_nn, X_train, y_train, cv=5)
    mlflow.log_metric("nn_cv_mean_accuracy", np.mean(cv_scores_nn))
    mlflow.log_metric("nn_cv_std_accuracy", np.std(cv_scores_nn))

    # Confusion matrix for NN
    cm_nn = confusion_matrix(y_test, y_pred_nn)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Neural Network Confusion Matrix')
    plt.savefig("Confusion-matrix-nn.png")
    mlflow.log_artifact("Confusion-matrix-nn.png")

    # Tags
    mlflow.set_tags({
        "Author": "Kashish",
        "Project": "Wine Classification",
        "Model": "NeuralNetwork",
        "Experiment_Date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "Scikit_Learn_Version": sklearn.__version__
    })

    # Log NN model
    mlflow.sklearn.log_model(best_nn, "NeuralNetwork-Model", input_example=input_example)
    

    print(f"NeuralNetwork Accuracy: {accuracy_nn}")