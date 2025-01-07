import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Train Linear Regression model
@st.cache_data
def train_model(data):
    # Define predictor variables (features) and target variable
    X = data[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Data Usage (MB/day)']]
    y = data['Battery Drain (mAh/day)']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Test the model
    y_pred = model.predict(X_test)
    metrics = {
        'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
        'Mean Squared Error': mean_squared_error(y_test, y_pred),
        'R2 Score': r2_score(y_test, y_pred)
    }
    
    return model, metrics, X_test, y_test, y_pred

# Predict Battery Drain
def predict_battery_drain(model, input_data):
    prediction = model.predict([input_data])
    return prediction[0]

# Streamlit App
st.title("Predict Battery Drain (mAh/day)")


# Load data
file_path = 'C:/Users/Dell/Desktop/prog/ids/new project/data/user_behavior_dataset.csv'
data = load_data(file_path)

# Train model
model, metrics, X_test, y_test, y_pred = train_model(data)

# Display Model Metrics
st.subheader("Model Performance Metrics")
for metric, value in metrics.items():
    st.write(f"{metric}: {value:.2f}")

# Display Graph of Actual vs Predicted Battery Drain
st.subheader("Actual vs Predicted Battery Drain")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.7)
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
ax.set_xlabel("Actual Battery Drain (mAh/day)")
ax.set_ylabel("Predicted Battery Drain (mAh/day)")
ax.set_title("Actual vs Predicted Battery Drain")
st.pyplot(fig)

# User Input Section
st.subheader("Provide Input Values for Prediction")
app_usage_time = st.number_input("App Usage Time (min/day):", min_value=0.0, step=1.0)
screen_on_time = st.number_input("Screen On Time (hours/day):", min_value=0.0, step=0.1)
data_usage = st.number_input("Data Usage (MB/day):", min_value=0, step=1)

# Predict Button
if st.button("Predict Battery Drain"):
    if app_usage_time > 0 and screen_on_time > 0 and data_usage > 0:
        input_data = [app_usage_time, screen_on_time, data_usage]
        prediction = predict_battery_drain(model, input_data)
        st.success(f"Predicted Battery Drain: {prediction:.2f} mAh/day")
    else:
        st.error("Please provide valid input values.")

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Load dataset
# @st.cache_data
# def load_data(file_path):
#     data = pd.read_csv(file_path)
#     return data

# # Train Linear Regression model
# @st.cache_data
# def train_model(data):
#     # Define predictor variables (features) and target variable
#     X = data[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Data Usage (MB/day)']]
#     y = data['Battery Drain (mAh/day)']
    
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Train the Linear Regression model
#     model = LinearRegression()
#     model.fit(X_train, y_train)
    
#     # Test the model
#     y_pred = model.predict(X_test)
#     metrics = {
#         'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
#         'Mean Squared Error': mean_squared_error(y_test, y_pred),
#         'R2 Score': r2_score(y_test, y_pred)
#     }
    
#     return model, metrics

# # Predict Battery Drain
# def predict_battery_drain(model, input_data):
#     prediction = model.predict([input_data])
#     return prediction[0]

# # Streamlit App
# st.title("Predict Battery Drain (mAh/day)")

# # Load data
# file_path = 'C:/Users/Dell/Desktop/prog/ids/new project/data/user_behavior_dataset.csv'
# data = load_data(file_path)

# # Train model
# model, metrics = train_model(data)

# # Display Model Metrics
# st.subheader("Model Performance Metrics")
# for metric, value in metrics.items():
#     st.write(f"{metric}: {value:.2f}")

# # User Input Section
# st.subheader("Provide Input Values for Prediction")
# app_usage_time = st.number_input("App Usage Time (min/day):", min_value=0.0, step=1.0)
# screen_on_time = st.number_input("Screen On Time (hours/day):", min_value=0.0, step=0.1)
# data_usage = st.number_input("Data Usage (MB/day):", min_value=0, step=1)

# # Predict Button
# if st.button("Predict Battery Drain"):
#     if app_usage_time > 0 and screen_on_time > 0 and data_usage > 0:
#         input_data = [app_usage_time, screen_on_time, data_usage]
#         prediction = predict_battery_drain(model, input_data)
#         st.success(f"Predicted Battery Drain: {prediction:.2f} mAh/day")
#     else:
#         st.error("Please provide valid input values.")

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import matplotlib.pyplot as plt

# # Load dataset
# @st.cache_data
# def load_data(file_path):
#     data = pd.read_csv(file_path)
#     return data

# # Train Linear Regression model
# @st.cache_data
# def train_model(data):
#     # Define predictor variables (features) and target variable
#     X = data[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Data Usage (MB/day)']]
#     y = data['Battery Drain (mAh/day)']
    
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Train the Linear Regression model
#     model = LinearRegression()
#     model.fit(X_train, y_train)
    
#     # Test the model
#     y_pred = model.predict(X_test)
#     metrics = {
#         'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
#         'Mean Squared Error': mean_squared_error(y_test, y_pred),
#         'R2 Score': r2_score(y_test, y_pred)
#     }
    
#     return model, metrics, X_test, y_test, y_pred

# # Predict Battery Drain
# def predict_battery_drain(model, input_data):
#     prediction = model.predict([input_data])
#     return prediction[0]

# # Streamlit App
# st.title("Predict Battery Drain (mAh/day)")

# # Load data
# file_path = 'C:/Users/Dell/Desktop/prog/ids/new project/data/user_behavior_dataset.csv'
# data = load_data(file_path)

# # Train model
# model, metrics, X_test, y_test, y_pred = train_model(data)

# # Display Model Metrics
# st.subheader("Model Performance Metrics")
# for metric, value in metrics.items():
#     st.write(f"{metric}: {value:.2f}")

# # User Input Section
# st.subheader("Provide Input Values for Prediction")
# app_usage_time = st.number_input("App Usage Time (min/day):", min_value=0.0, step=1.0)
# screen_on_time = st.number_input("Screen On Time (hours/day):", min_value=0.0, step=0.1)
# data_usage = st.number_input("Data Usage (MB/day):", min_value=0, step=1)

# # Initialize variables to store user prediction
# user_prediction = None
# input_data = None

# # Predict Button
# if st.button("Predict Battery Drain"):
#     if app_usage_time > 0 and screen_on_time > 0 and data_usage > 0:
#         input_data = [app_usage_time, screen_on_time, data_usage]
#         user_prediction = predict_battery_drain(model, input_data)
#         st.success(f"Predicted Battery Drain: {user_prediction:.2f} mAh/day")
#     else:
#         st.error("Please provide valid input values.")

# # Display Graph of Actual vs Predicted Battery Drain
# st.subheader("Actual vs Predicted Battery Drain")
# fig, ax = plt.subplots()
# ax.scatter(y_test, y_pred, alpha=0.7, label="Test Data Predictions")
# ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label="Perfect Fit Line")
# ax.set_xlabel("Actual Battery Drain (mAh/day)")
# ax.set_ylabel("Predicted Battery Drain (mAh/day)")
# ax.set_title("Actual vs Predicted Battery Drain")

# # Highlight user prediction on the graph
# if user_prediction is not None and input_data is not None:
#     ax.scatter([user_prediction], [user_prediction], color='green', s=100, label="Your Prediction")
#     ax.annotate("Your Prediction", (user_prediction, user_prediction), textcoords="offset points", xytext=(0, 10), ha='center')

# ax.legend()
# st.pyplot(fig)
