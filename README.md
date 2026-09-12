# ⚡ Energy AI – Energy Consumption Forecasting

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning web application that forecasts electricity consumption using real-world household electricity data.

🔗 **Live Demo:** https://energy-forecasting-abhi.streamlit.app/

---

## 📌 Project Overview

Energy consumption changes throughout the day and can be difficult to estimate manually.

This project uses **Machine Learning** to analyze historical electricity consumption and forecast future energy usage.

The application provides an interactive dashboard where users can explore energy consumption patterns and generate forecasts.

---

## 🎯 Objectives

- Analyze real-world electricity consumption data
- Preprocess and transform energy consumption data
- Build a Machine Learning forecasting model
- Predict future electricity consumption
- Visualize actual and predicted energy usage
- Provide an interactive web application
- Generate a 24-hour energy consumption forecast

---

## 🤖 Machine Learning Model

### Random Forest Regression

The project uses **Random Forest Regression** for energy consumption forecasting.

Random Forest combines multiple decision trees and produces a prediction by combining their individual results.

**Why Random Forest?**

- Handles nonlinear relationships
- Works well with numerical features
- Robust against overfitting compared with a single decision tree
- Can capture complex patterns in historical consumption data

---

## 📊 Dataset

The project uses the **Individual Household Electric Power Consumption Dataset** from the UCI Machine Learning Repository.

The original dataset contains electricity measurements collected at one-minute intervals from a household.

For this project, the electricity data was processed and converted into **hourly energy consumption**.

**Temperature data is not used in this project.**

---

## 🔧 Features Used

The model uses the following features:

- Hour
- Day of Week
- Month
- Lag 1 Hour
- Lag 24 Hours
- Lag 168 Hours
- Rolling 24-Hour Consumption

These historical and time-based features help the model identify consumption patterns.

---

## 🛠️ Technologies Used

### Programming
- Python

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- Random Forest Regression

### Visualization
- Plotly

### Web Application
- Streamlit

### Version Control
- Git
- GitHub

---

## 📈 Model Performance

The application evaluates the model using:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

Current application result:

| Metric | Result |
|---|---:|
| MAE | 0.335 kWh |
| RMSE | 0.487 kWh |
| R² Score | 0.574 |

The results are based on the project's chronological train/test evaluation.

---

## 🚀 Application Features

### 📊 Dashboard
Interactive overview of the electricity consumption dataset.

### 📈 Energy Trend
Visualizes historical energy consumption patterns.

### 🎯 Actual vs Predicted
Compares actual energy consumption with model predictions.

### 🔮 Energy Forecast
Allows users to generate energy consumption predictions.

### ⏰ Next 24-Hour Forecast
Generates a forecast for the next 24 hours.

### 📥 Download Results
Forecast results can be downloaded as a CSV file.

---

## 🏗️ Project Structure

```text
Energy_Forecasting/
│
├── app.py
├── energy_data.csv
├── generate_data.py
├── process_real_data.py
├── requirements.txt
├── .gitignore
└── README.md
