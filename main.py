import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Data Generation (Simulating a real-world classroom environment)
def generate_student_data(n=500):
    np.random.seed(42)
    data = {
        'Study_Hours_Weekly': np.random.randint(5, 30, n),
        'Attendance_Rate': np.random.uniform(60, 100, n),
        'Previous_Grades': np.random.uniform(40, 100, n),
        'Internet_Access': np.random.choice([0, 1], n),
        'Extra_Curricular': np.random.choice([0, 1], n),
    }
    df = pd.DataFrame(data)
    
    # Target: 1 if student passes (Grade > 50), 0 otherwise
    # Logic: Grades are influenced by study hours and attendance
    noise = np.random.normal(0, 5, n)
    score = (df['Study_Hours_Weekly'] * 1.2) + (df['Attendance_Rate'] * 0.5) + noise
    df['Pass'] = (score > 65).astype(int)
    return df

# 2. Load and Explore
df = generate_student_data()
print("Dataset Head:\n", df.head())

# 3. Preprocessing
X = df.drop('Pass', axis=1)
y = df['Pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation
predictions = model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, predictions))

# 6. Feature Importance Visualization
importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title('Which factors most affect student success?')
plt.show()