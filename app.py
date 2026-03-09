import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# Dataset (embedded)
# ---------------------------

data = {
"sleep":[8,7,6,5,4,7,6,5,4,3,8,6,5,4,7,6,5,3],
"study":[6,5,4,3,2,6,4,3,2,1,7,5,3,2,6,4,3,1],
"nervous":[0,0,1,1,1,0,1,1,1,1,0,0,1,1,0,1,1,1],
"heart":[0,0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1],
"focus":[0,0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1],
"confidence":[8,7,6,4,3,8,6,4,3,2,9,7,5,3,8,6,4,2],
"tests":[5,4,3,2,1,5,3,2,1,0,6,4,2,1,5,3,2,0],
"anxiety":["Low","Low","Medium","High","High","Low","Medium","High","High","High","Low","Medium","High","High","Low","Medium","High","High"]
}

df = pd.DataFrame(data)

X = df.drop("anxiety",axis=1)
y = df["anxiety"]

# split data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# ---------------------------
# Train Models
# ---------------------------

dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression(max_iter=1000)

dt.fit(X_train,y_train)
rf.fit(X_train,y_train)
lr.fit(X_train,y_train)

# accuracy
dt_acc = accuracy_score(y_test,dt.predict(X_test))
rf_acc = accuracy_score(y_test,rf.predict(X_test))
lr_acc = accuracy_score(y_test,lr.predict(X_test))

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("AI Exam Anxiety Detector")

st.write("Enter student details:")

sleep = st.slider("Sleep Hours",0,10,6)
study = st.slider("Study Hours",0,10,4)
nervous = st.selectbox("Nervous before exam?",["No","Yes"])
heart = st.selectbox("Heart beating fast?",["No","Yes"])
focus = st.selectbox("Difficulty concentrating?",["No","Yes"])
confidence = st.slider("Confidence Level",1,10,5)
tests = st.slider("Mock Tests Taken",0,10,3)

# convert yes/no
nervous = 1 if nervous=="Yes" else 0
heart = 1 if heart=="Yes" else 0
focus = 1 if focus=="Yes" else 0

input_data = [[sleep,study,nervous,heart,focus,confidence,tests]]

# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict Anxiety"):

    prediction = rf.predict(input_data)

    st.subheader("Predicted Anxiety Level:")
    st.success(prediction[0])

    if prediction[0]=="High":
        st.warning("Suggestion: Take breaks and practice relaxation.")
    elif prediction[0]=="Medium":
        st.info("Suggestion: Revise calmly and maintain sleep.")
    else:
        st.success("You seem relaxed. Keep preparing.")

# ---------------------------
# Model Accuracy Comparison
# ---------------------------

st.subheader("Model Accuracy Comparison")

acc_data = pd.DataFrame({
"Model":["Decision Tree","Random Forest","Logistic Regression"],
"Accuracy":[dt_acc,rf_acc,lr_acc]
})

st.write(acc_data)

fig = plt.figure()
plt.bar(acc_data["Model"],acc_data["Accuracy"])
plt.ylabel("Accuracy")
plt.title("Model Performance")

st.pyplot(fig)