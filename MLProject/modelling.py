import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def main():
    data_path = "heart_disease_preprocessing.csv"
    df = pd.read_csv(data_path)

    target_col = "target"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.autolog()  

    with mlflow.start_run(run_name="logreg_basic"):
        model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs"
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy:", acc)
        print("F1-score:", f1)


        mlflow.log_metric("accuracy_manual", acc)
        mlflow.log_metric("f1_manual", f1)

        mlflow.sklearn.log_model(model, artifact_path="model")

if __name__ == "__main__":
    main()