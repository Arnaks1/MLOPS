import pandas as pd
import os


def clean_data():
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv("data/raw/titanic.csv")

    # Удаление ненужных столбцов
    df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    # Обработка пропусков
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Замена категориальных признаков
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Добавление нового признака
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df.to_csv('data/processed/cleaned_titanic.csv', index=False)


if __name__ == "__main__":
    clean_data()
