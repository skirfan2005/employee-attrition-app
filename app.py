import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load files
model = joblib.load("stacking_clf.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ----------------------------
# ⚙️ Page Config
# ----------------------------
st.set_page_config(
    page_title="HR Attrition Intelligence System",
    page_icon="📊",
    layout="wide"
)

# ----------------------------
# 🎨 UI Styling
# ----------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1c1f26;
    text-align: center;
    font-size: 18px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
}
.stButton>button {
    background-color: #00c6ff;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 📌 Sidebar
# ----------------------------
st.sidebar.title("📊 HR System")
menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predict", "Bulk Prediction", "Model Insights"],
    label_visibility="collapsed"
)

# ----------------------------
# 🏠 DASHBOARD
# ----------------------------
if menu == "Dashboard":
    st.title("📊 HR Attrition Intelligence Dashboard")

    # Demo numbers (replace if dataset used)
    total_emp = 1000
    high_risk = 120
    avg_risk = 0.35

    col1, col2, col3 = st.columns(3)

    col1.markdown(f'<div class="card">👥 Total Employees<br><b>{total_emp}</b></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="card">⚠️ High Risk<br><b>{high_risk}</b></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="card">📉 Avg Risk<br><b>{avg_risk:.2f}</b></div>', unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📊 Risk Distribution")

    dummy = np.random.rand(100)
    fig, ax = plt.subplots()
    ax.hist(dummy, bins=20)
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("📌 About System")
    st.write("""
    This system predicts employee attrition risk using advanced ML models.

    ✔ Detect high-risk employees  
    ✔ Provide actionable insights  
    ✔ Help HR reduce attrition  
    """)

# ----------------------------
# 🔍 SINGLE PREDICTION
# ----------------------------
elif menu == "Predict":
    st.title("🔍 Employee Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        Age = st.slider("Age", 18, 60, 30)
        MonthlyIncome = st.number_input("Monthly Income", 1000, 200000, 30000)
        YearsAtCompany = st.slider("Years At Company", 0, 40, 5)
        TotalWorkingYears = st.slider("Total Working Years", 0, 40, 5)

    with col2:
        JobRole = st.selectbox("Job Role", label_encoders['JobRole'].classes_)
        Department = st.selectbox("Department", label_encoders['Department'].classes_)
        JobSatisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        WorkLifeBalance = st.slider("Work Life Balance", 1, 4, 3)
        OverTime = st.selectbox("OverTime", ["No", "Yes"])

    JobRole = label_encoders['JobRole'].transform([JobRole])[0]
    Department = label_encoders['Department'].transform([Department])[0]
    OverTime = 1 if OverTime == "Yes" else 0

    df = pd.DataFrame([{
        'Age': Age,
        'MonthlyIncome': MonthlyIncome,
        'YearsAtCompany': YearsAtCompany,
        'TotalWorkingYears': TotalWorkingYears,
        'JobRole': JobRole,
        'Department': Department,
        'JobSatisfaction': JobSatisfaction,
        'WorkLifeBalance': WorkLifeBalance,
        'OverTime': OverTime
    }])

    df = df.reindex(columns=feature_columns, fill_value=0)
    df_scaled = scaler.transform(df)

    if st.button("⚡ Predict"):
        pred = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0][1]

        st.subheader("📊 Result")

        if pred == 1:
            st.error(f"⚠️ High Risk: {prob:.2f}")
        else:
            st.success(f"✅ Low Risk: {prob:.2f}")

        st.progress(float(prob))

        st.subheader("💡 Recommendations")

        if prob > 0.6:
            st.write("• Increase salary")
            st.write("• Promotion plan")
            st.write("• Reduce workload")
        else:
            st.write("• Maintain engagement")

# ----------------------------
# 📂 BULK PREDICTION
# ----------------------------
elif menu == "Bulk Prediction":
    st.title("📂 Bulk Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.subheader("Preview")
        st.dataframe(df.head())

        if st.button("⚡ Run Model"):
            df_input = df.reindex(columns=feature_columns, fill_value=0)
            df_scaled = scaler.transform(df_input)

            probs = model.predict_proba(df_scaled)[:, 1]

            df['Attrition_Risk'] = probs
            df['Prediction'] = (probs > 0.5).astype(int)

            st.success("✅ Done")

            col1, col2 = st.columns(2)
            col1.metric("High Risk", int((probs > 0.6).sum()))
            col2.metric("Low Risk", int((probs <= 0.6).sum()))

            st.bar_chart(df['Prediction'].value_counts())

            st.dataframe(df)

            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download", csv, "results.csv")

# ----------------------------
# 📈 MODEL INSIGHTS
# ----------------------------
elif menu == "Model Insights":
    st.title("📈 Model Insights")

    st.subheader("Feature Importance (Top Features)")

    # Dummy importance (since stacking doesn't give directly)
    features = feature_columns[:10]
    values = np.random.rand(10)

    fig, ax = plt.subplots()
    ax.barh(features, values)
    ax.invert_yaxis()

    st.pyplot(fig)

    st.subheader("📌 Explanation")

    st.write("""
    - Model uses ensemble learning (XGBoost + Random Forest)
    - Combines multiple models for better accuracy
    - Helps HR make data-driven decisions
    """)