
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

st.set_page_config(layout="wide")
st.title("AI Driven Personal Loan Marketing Dashboard")

data = pd.read_csv("UniversalBank.csv")

st.header("Descriptive Analytics")

col1,col2,col3 = st.columns(3)

with col1:
    fig = px.histogram(data, x="Age", title="Age Distribution")
    st.plotly_chart(fig)
    st.caption("Shows the age spread of customers to understand life stage segments.")

with col2:
    fig = px.histogram(data, x="Income", title="Income Distribution")
    st.plotly_chart(fig)
    st.caption("Higher income customers generally have higher loan eligibility.")

with col3:
    fig = px.box(data, x="Education", y="Income", title="Income by Education Level")
    st.plotly_chart(fig)
    st.caption("Education level often correlates with income potential.")

loan_rate = data["Personal Loan"].value_counts(normalize=True)*100
st.subheader("Loan Acceptance Rate (%)")
st.write(loan_rate)

X = data.drop(columns=["Personal Loan","ID","ZIPCode"])
y = data["Personal Loan"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

models = {
"Decision Tree": DecisionTreeClassifier(),
"Random Forest": RandomForestClassifier(),
"Gradient Boosting": GradientBoostingClassifier()
}

results=[]
roc_data={}
conf_mats={}

for name,model in models.items():
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    probs=model.predict_proba(X_test)[:,1]

    results.append({
        "Model":name,
        "Train Accuracy":model.score(X_train,y_train),
        "Test Accuracy":accuracy_score(y_test,preds),
        "Precision":precision_score(y_test,preds),
        "Recall":recall_score(y_test,preds),
        "F1 Score":f1_score(y_test,preds)
    })

    fpr,tpr,_=roc_curve(y_test,probs)
    roc_data[name]=(fpr,tpr,auc(fpr,tpr))

    conf_mats[name]=confusion_matrix(y_test,preds)

st.header("Model Performance Comparison")
st.dataframe(pd.DataFrame(results))

st.header("ROC Curve")
fig,ax = plt.subplots()

for name,(fpr,tpr,roc_auc) in roc_data.items():
    ax.plot(fpr,tpr,label=f"{name} AUC={roc_auc:.2f}")

ax.plot([0,1],[0,1],'--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()

st.pyplot(fig)

st.header("Confusion Matrices")

for name,cm in conf_mats.items():
    st.subheader(name)
    st.write(cm)

rf = RandomForestClassifier()
rf.fit(X,y)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

st.header("Feature Importance (Marketing Insights)")
fig = px.bar(importances, title="Key Drivers of Personal Loan Acceptance")
st.plotly_chart(fig)
st.caption("Features with higher importance influence the loan acceptance prediction more strongly.")

st.header("Predict Personal Loan Acceptance")

uploaded = st.file_uploader("Upload CSV for prediction")

if uploaded:
    new_data = pd.read_csv(uploaded)
    new_data = new_data.drop(columns=["ID","ZIPCode"], errors="ignore")
    new_data = new_data[X.columns]

    preds = rf.predict(new_data)
    probs = rf.predict_proba(new_data)[:,1]

    new_data["Predicted Personal Loan"] = preds
    new_data["Acceptance Probability"] = probs

    st.dataframe(new_data)

    csv = new_data.to_csv(index=False).encode()
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
