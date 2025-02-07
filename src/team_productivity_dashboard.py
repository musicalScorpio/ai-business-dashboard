import streamlit as st
import pandas as pd
import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import plotly.express as px  # Import Plotly Express


# Streamlit UI setup
st.set_page_config(page_title="Team Productivity Dashboard", layout="wide")
st.title("📊 Team Productivity Dashboard & Prediction Engine")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file with Task Data", type=["csv"])


def load_csv(file):
    df = pd.read_csv(file, parse_dates=["start_date", "due_date", "actual_completion_date"])
    return df


def prepare_data(df):
    # Convert to datetime, handling errors
    df["start_date"] = pd.to_datetime(df["start_date"], format="%Y-%m-%d", errors='coerce')
    df["due_date"] = pd.to_datetime(df["due_date"], format="%Y-%m-%d", errors='coerce')

    df["task_duration"] = (df["due_date"] - df["start_date"]).dt.days
    df["actual_duration"] = (df["actual_completion_date"] - df["start_date"]).dt.days
    df["delay"] = df["actual_duration"] - df["task_duration"]
    df["delayed"] = df["delay"].apply(lambda x: 1 if x > 0 else 0)
    df = df.dropna()
    return df


def train_model(df):
    X = df[["task_duration"]]
    y = df["delay"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    error = mean_absolute_error(y_test, y_pred)

    return model, error


def predict_task_completion(model, start_date, due_date):
    task_duration = (due_date - start_date).days
    predicted_delay = model.predict(np.array([[task_duration]]))[0]
    predicted_completion = due_date + datetime.timedelta(days=int(predicted_delay))
    return predicted_completion, predicted_delay

if uploaded_file:
    df = load_csv(uploaded_file)
    df = prepare_data(df)

    model, error = train_model(df)

    st.subheader("📌 Task Overview")
    st.dataframe(df)

    st.write(f"📊 Model Training Complete - Mean Absolute Error: {error:.2f} days")

    # --- Scatter Plots ---
    st.subheader("📈 Task Duration vs. Delay")
    fig_duration_delay = px.scatter(
        df, x="task_duration", y="delay",
        title="Task Duration vs. Delay",
        labels={"task_duration": "Task Duration (days)", "delay": "Delay (days)"}
    )
    st.plotly_chart(fig_duration_delay, use_container_width=True)

    st.subheader("📅 Start Date vs. Delay")
    fig_start_date_delay = px.scatter(
        df, x="start_date", y="delay",
        title="Start Date vs. Delay",
        labels={"start_date": "Start Date", "delay": "Delay (days)"}
    )
    st.plotly_chart(fig_start_date_delay, use_container_width=True)

    # --- Prediction Engine ---
    st.subheader("🔮 Predict New Task Completion")
    new_start_date = st.date_input("Start Date")
    new_due_date = st.date_input("Due Date")

    if st.button("Predict Task Completion Date"):
        predicted_completion, predicted_delay = predict_task_completion(model, new_start_date, new_due_date)
        st.write(f"Predicted Completion Date: {predicted_completion.strftime('%Y-%m-%d')}")
        st.write(f"Expected Delay: {predicted_delay:.2f} days")


