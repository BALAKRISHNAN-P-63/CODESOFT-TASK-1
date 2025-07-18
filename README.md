# CODESOFT-TASK-1
🚢 Titanic Survival Prediction A beginner-friendly machine learning project using the Titanic dataset to predict passenger survival based on features like age, gender, and class. Built with Python, pandas, scikit-learn, and visualized with seaborn/matplotlib.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
# Load Titanic dataset
df = pd.read_csv('Titanic-Dataset.csv')
df.head()
# Check missing values
df.isnull().sum()
# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop unnecessary columns (safely)
columns_to_drop = ['Cabin', 'Ticket', 'Name', 'PassengerId']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

print("Missing values handled and unnecessary columns dropped. Here's the cleaned data:")
print(df.head())
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])        # male = 1, female = 0
df['Embarked'] = le.fit_transform(df['Embarked'])  # S=2, C=0, Q=1 (example mapping)

print("Categorical columns encoded. Here's the data preview:")
print(df.head())
X = df.drop('Survived', axis=1)
y = df['Survived']

print("Features and target variable defined.")
print("Feature columns:", list(X.columns))
print("Target column: Survived")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data split into training and testing sets.")
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Random Forest model trained successfully.")
importances = pd.Series(model.feature_importances_, index=X.columns)

print("Feature importances:")
print(importances.sort_values(ascending=False))

# Plot in pink
importances.nlargest(10).plot(kind='barh', figsize=(8, 5), color='pink')
plt.title("Feature Importances", fontsize=14)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()
sns.countplot(x='Embarked', hue='Survived', data=df, palette='magma')
plt.title("Survival by Port of Embarkation")
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()
sns.countplot(x='Pclass', hue='Survived', data=df, palette='coolwarm')
plt.title("Survival by Passenger Class")
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()
sns.countplot(x='Sex', hue='Survived', data=df, palette='Set2')
plt.title("Survival by Gender")
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()
plt.figure(figsize=(10, 5))
sns.histplot(df['Fare'], bins=40, kde=True, color='k')
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()
