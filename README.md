# AI-Based-Clinical-Decision-System-Chatbot-for-Diabetes-Management
A healthcare-focused AI project that combines a diabetes information chatbot with a Machine Learning-based risk screening system. Built using Python and Streamlit, the application uses a Random Forest Classifier to analyze health parameters and provide basic diabetes risk predictions.
# AI-Based Clinical Decision Support Chatbot for Diabetes Management

## About the Project

Diabetes is a common health condition, and people often have questions about its symptoms, risk factors, diet, exercise, HbA1c, and complications. This project was developed as a simple Clinical Decision Support prototype to provide useful diabetes-related information and basic risk screening in one application.

The project has two main parts:

1. **Diabetes Information Chatbot** – Users can ask questions related to diabetes and get simple educational responses.
2. **Diabetes Risk Screening** – Users can enter health parameters such as glucose, BMI, blood pressure, age, and other values. A Machine Learning model then provides a basic diabetes-risk prediction.

## How It Works

The application is built using Streamlit, which provides the web interface.

For risk screening, the entered health information is processed and passed to a **Random Forest Classifier**. The model uses patterns learned from the available demonstration data to generate the prediction.

The chatbot uses structured Python-based responses to answer common diabetes-related questions.

### Basic workflow

```text
User
 ↓
Streamlit Interface
 ↓
 ┌─────────────────────┐
 │                     │
Chatbot          Risk Screening
 │                     │
 ↓                     ↓
Response          Random Forest
                       ↓
                 Risk Prediction
