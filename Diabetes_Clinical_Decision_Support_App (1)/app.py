
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(
    page_title="Diabetes Clinical Decision Support Chatbot",
    page_icon="🩺",
    layout="wide"
)

# -------------------- MODEL --------------------
@st.cache_resource
def train_model():
    # Synthetic demonstration data so the app works immediately.
    # Replace with an approved public diabetes dataset for final research evaluation.
    rng = np.random.default_rng(42)
    n = 1500

    pregnancies = rng.poisson(2.5, n).clip(0, 15)
    glucose = rng.normal(120, 30, n).clip(50, 240)
    blood_pressure = rng.normal(72, 12, n).clip(40, 130)
    skin = rng.normal(23, 9, n).clip(5, 70)
    insulin = rng.lognormal(4.5, 0.7, n).clip(10, 600)
    bmi = rng.normal(29, 6, n).clip(16, 55)
    dpf = rng.gamma(2, 0.25, n).clip(0.05, 2.5)
    age = rng.normal(35, 11, n).clip(18, 80)

    score = (
        -7
        + 0.035 * glucose
        + 0.055 * bmi
        + 0.018 * age
        + 0.10 * pregnancies
        + 0.012 * blood_pressure
        + 0.9 * dpf
    )

    probability = 1 / (1 + np.exp(-score))
    outcome = rng.binomial(1, probability)

    X = pd.DataFrame({
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    })
    y = outcome

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob)
    }

    return model, metrics

model, metrics = train_model()

# -------------------- CHATBOT --------------------
def chatbot_response(question):
    q = question.lower().strip()

    # Emergency safety check first
    emergency = [
        "unconscious", "passed out", "seizure", "convulsion",
        "chest pain", "difficulty breathing", "can't breathe",
        "severe confusion", "fainting", "severe vomiting"
    ]

    if any(word in q for word in emergency):
        return """🚨 **Please seek urgent medical care.**

The symptom you described may require immediate professional assessment.
Do not rely on this chatbot for emergency diagnosis or treatment.

If this is an emergency, contact your local emergency service or go to the nearest emergency department."""

    if any(x in q for x in ["hello", "hi", "hey"]):
        return """Hello! 👋 I'm your **Diabetes Clinical Decision Support Chatbot**.

You can ask me about:
- Diabetes and its types
- Symptoms and risk factors
- HbA1c
- Diet and lifestyle
- Exercise
- Diabetes complications
- Common diabetes medicines
- When to seek medical help"""

    if "what is diabetes" in q or "define diabetes" in q:
        return """### What is diabetes?

Diabetes is a group of conditions in which blood glucose levels become too high because the body does not produce enough insulin, does not use insulin effectively, or both.

The major types include **Type 1 diabetes, Type 2 diabetes, and gestational diabetes**.

A chatbot cannot diagnose diabetes. Diagnosis requires appropriate clinical assessment and laboratory testing."""

    if any(x in q for x in ["symptom", "signs"]):
        return """### Common symptoms of diabetes

Possible symptoms include:
- Increased thirst
- Frequent urination
- Increased hunger
- Unexplained weight changes
- Tiredness
- Blurred vision
- Slow-healing wounds

Some people, particularly those with Type 2 diabetes, may have few noticeable symptoms.

If symptoms persist or concern you, consult a healthcare professional."""

    if "hba1c" in q or "a1c" in q:
        return """### HbA1c

HbA1c is a blood test that reflects average blood glucose over approximately the previous 2–3 months.

Healthcare professionals use it as one part of diabetes diagnosis and monitoring.

The appropriate target varies between individuals, so do not change treatment based only on an HbA1c value."""

    if any(x in q for x in ["diet", "food", "eat", "nutrition", "meal"]):
        return """### General diabetes nutrition guidance

A balanced eating pattern can include:
- Vegetables
- Whole grains and high-fibre foods
- Appropriate portions of protein
- Appropriate portions of carbohydrates
- Water instead of sugar-sweetened drinks

Limit highly refined carbohydrates and sugary drinks.

There is no single diet suitable for everyone. A dietitian or healthcare professional can personalize nutrition advice."""

    if any(x in q for x in ["exercise", "workout", "physical activity"]):
        return """### Physical activity

Regular physical activity can support blood-glucose management and overall health.

The type and intensity of exercise should be appropriate for your health, medications, fitness level, and any diabetes-related complications.

If you are starting a new exercise program or have medical complications, discuss a safe plan with your healthcare professional."""

    if any(x in q for x in ["medicine", "medication", "drug", "metformin", "insulin"]):
        return """### Diabetes medications

Diabetes treatment may include lifestyle measures and medicines such as metformin, insulin, and other medication classes.

The appropriate medicine depends on factors such as diabetes type, glucose levels, kidney function, other health conditions, pregnancy status, and individual treatment goals.

⚠️ **Do not start, stop, or change a medicine or dose based on this chatbot.** Medication decisions should be made with a qualified healthcare professional."""

    if any(x in q for x in ["risk factor", "risk factors", "family history", "risk of diabetes"]):
        return """### Risk factors

Risk can be influenced by:
- Family history
- Increasing age
- Overweight/obesity
- Physical inactivity
- Previous gestational diabetes
- Certain medical conditions

Having a risk factor does not mean a person will definitely develop diabetes.

A healthcare professional can advise whether diabetes screening is appropriate."""

    if any(x in q for x in ["prevent", "prevention"]):
        return """### Diabetes prevention

For people at increased risk of Type 2 diabetes, healthy eating, regular physical activity, appropriate weight management, adequate sleep, and regular medical follow-up can help reduce risk.

Prevention advice should be personalized to the individual."""

    if any(x in q for x in ["complication", "kidney", "eye", "nerve", "foot"]):
        return """### Diabetes complications

Persistently high blood glucose can contribute to complications affecting:
- Eyes
- Kidneys
- Nerves
- Heart and blood vessels
- Feet

Regular monitoring and appropriate healthcare can help reduce the risk of complications.

New severe symptoms should be assessed promptly by a healthcare professional."""

    if any(x in q for x in ["low sugar", "low blood sugar", "hypoglycemia", "hypo"]):
        return """### Possible low blood sugar

Low blood glucose can cause symptoms such as:
- Shakiness
- Sweating
- Hunger
- Dizziness
- Weakness
- Confusion

If you have diabetes, follow the hypoglycemia plan provided by your healthcare professional.

If the person is unconscious, having a seizure, severely confused, or unable to safely swallow, seek emergency medical help."""

    return """I can help with general diabetes information.

Try asking:
- **What is diabetes?**
- **What are the symptoms of diabetes?**
- **What is HbA1c?**
- **What foods are suitable for diabetes management?**
- **What are common diabetes medicines?**
- **What are the risk factors for diabetes?**
- **What are diabetes complications?**

For diagnosis or treatment decisions, consult a qualified healthcare professional."""

# -------------------- UI --------------------
st.markdown(
    "<h1 style='text-align:center;'>🩺 Diabetes Clinical Decision Support Chatbot</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>AI-assisted diabetes education and risk-screening prototype</p>",
    unsafe_allow_html=True
)

st.warning(
    "⚠️ Student research prototype only. This application does not diagnose disease, "
    "prescribe medication, or replace a qualified healthcare professional."
)

tab1, tab2, tab3 = st.tabs([
    "💬 AI Chatbot",
    "📊 Diabetes Risk Screening",
    "📈 Model Performance"
])

# -------------------- CHAT TAB --------------------
with tab1:
    st.subheader("Ask the Diabetes Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a diabetes-related question...")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        answer = chatbot_response(question)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# -------------------- RISK TAB --------------------
with tab2:
    st.subheader("Diabetes Risk Screening")

    st.info(
        "Enter the values below to obtain a model-based screening estimate. "
        "The result is not a medical diagnosis."
    )

    c1, c2 = st.columns(2)

    with c1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose (mg/dL)", 0.0, 300.0, 120.0)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", 0.0, 180.0, 72.0)
        skin = st.number_input("Skin Thickness (mm)", 0.0, 100.0, 20.0)

    with c2:
        insulin = st.number_input("Insulin (µU/mL)", 0.0, 900.0, 80.0)
        bmi = st.number_input("BMI", 10.0, 80.0, 25.0, step=0.1)
        dpf = st.number_input(
            "Diabetes Pedigree Function", 0.0, 3.0, 0.47, step=0.01
        )
        age = st.number_input("Age", 18, 120, 30)

    if st.button("🔍 Calculate Risk", type="primary"):
        data = pd.DataFrame([[
            pregnancies,
            glucose,
            blood_pressure,
            skin,
            insulin,
            bmi,
            dpf,
            age
        ]], columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ])

        probability = model.predict_proba(data)[0][1]
        percentage = probability * 100

        if percentage < 35:
            st.success(f"Lower model-estimated risk: {percentage:.1f}%")
        elif percentage < 65:
            st.warning(f"Intermediate model-estimated risk: {percentage:.1f}%")
        else:
            st.error(f"Higher model-estimated risk: {percentage:.1f}%")

        st.progress(float(probability))

        st.info(
            "This is a machine-learning screening estimate from a student prototype. "
            "It must not be interpreted as confirmation or exclusion of diabetes. "
            "Clinical testing and professional assessment are required."
        )

# -------------------- PERFORMANCE TAB --------------------
with tab3:
    st.subheader("Machine Learning Model Performance")

    metric_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": [round(v, 4) for v in metrics.values()]
    })

    st.dataframe(metric_df, use_container_width=True, hide_index=True)

    st.caption(
        "These values are from the synthetic demonstration dataset used to make the "
        "prototype runnable immediately. For your final research report, use an "
        "approved public dataset and report the resulting evaluation separately."
    )

st.caption("AI-Based Clinical Decision Support Chatbot for Diabetes Management")
