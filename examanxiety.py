import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("dataset.csv")

# Features and labels
X = data[['sleep','study','nervous','heart','focus']]
y = data['anxiety']

# Train model
model = DecisionTreeClassifier()
model.fit(X,y)

print("----- Exam Anxiety Detector -----")

# User input
sleep = int(input("Hours of sleep (0-10): "))
study = int(input("Study hours per day (0-10): "))

nervous = input("Do you feel nervous before exam? (yes/no): ")
heart = input("Heart beating fast before exam? (yes/no): ")
focus = input("Difficulty concentrating? (yes/no): ")

# Convert yes/no to 1/0
nervous = 1 if nervous.lower()=="yes" else 0
heart = 1 if heart.lower()=="yes" else 0
focus = 1 if focus.lower()=="yes" else 0

# Prediction
prediction = model.predict([[sleep,study,nervous,heart,focus]])

print("\nPredicted Anxiety Level:", prediction[0])

if prediction[0] == "High":
    print("Suggestion: Take short breaks, practice deep breathing.")
elif prediction[0] == "Medium":
    print("Suggestion: Revise calmly and maintain sleep schedule.")
else:
    print("Suggestion: You seem relaxed. Keep preparing steadily.")