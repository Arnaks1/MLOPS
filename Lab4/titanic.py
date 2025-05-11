from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import pickle

default_args = {
    'start_date': datetime(2025, 2, 3),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id="titanic_train_pipeline",
    default_args=default_args,
    schedule=timedelta(hours=1),
    catchup=False,
    tags=["ml", "titanic"],
) as dag:

    def load_and_clean_data():
        df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

        cat_columns = ['Name', 'Ticket', 'Cabin', 'Embarked']
        num_columns = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
        df['Cabin'] = df['Cabin'].fillna('Unknown')

        df = df.reset_index(drop=True)

        ordinal = OrdinalEncoder()
        df[cat_columns] = ordinal.fit_transform(df[cat_columns])

        df.to_csv('/tmp/titanic_clean.csv', index=False)

    def preprocess_data():
        df = pd.read_csv('/tmp/titanic_clean.csv')
        X = df.drop(columns=['Survived'])
        y = df['Survived']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Сохраняем для последующих шагов
        with open('/tmp/train_data.pkl', 'wb') as f:
            pickle.dump((X_train, X_val, y_train, y_val), f)

    def train_and_log_model():
        with open('/tmp/train_data.pkl', 'rb') as f:
            X_train, X_val, y_train, y_val = pickle.load(f)

        input_example = X_train[:5]

        with mlflow.start_run(run_name="LogisticRegression"):
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred)
            rec = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)

            mlflow.log_param("model", "LogisticRegression")
            mlflow.log_param("max_iter", 1000)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(model, "model", input_example=input_example)

    load_task = PythonOperator(
        task_id="load_and_clean_data",
        python_callable=load_and_clean_data
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data
    )

    train_task = PythonOperator(
        task_id="train_and_log_model",
        python_callable=train_and_log_model
    )

    load_task >> preprocess_task >> train_task
