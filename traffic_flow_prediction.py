# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import math
import warnings

warnings.filterwarnings("ignore")

# Load Dataset
data = pd.read_csv("/content/drive/MyDrive/Dataset/traffic.csv")
data["DateTime"] = pd.to_datetime(data["DateTime"])
data = data.drop(["ID"], axis=1)

# Data Information
print(data.info())

# Data Copy
df = data.copy()

# Plot Traffic on Junctions Over Time
colors = ["#FFD4DB", "#BBE7FE", "#D3B5E5", "#dfe2b6"]
plt.figure(figsize=(20, 6), facecolor="#627D78")
Time_series = sns.lineplot(data=df, x="DateTime", y="Vehicles", hue="Junction", palette=colors)
Time_series.set_title("Traffic On Junctions Over Years")
Time_series.set_ylabel("Number of Vehicles")
Time_series.set_xlabel("Date")
plt.show()

# Feature Engineering
df["Year"] = df["DateTime"].dt.year
df["Month"] = df["DateTime"].dt.month
df["Date_no"] = df["DateTime"].dt.day
df["Hour"] = df["DateTime"].dt.hour
df["Day"] = df["DateTime"].dt.strftime("%A")

# Plot Features
new_features = ["Year", "Month", "Date_no", "Hour", "Day"]
for feature in new_features:
    plt.figure(figsize=(10, 2), facecolor="#627D78")
    ax = sns.lineplot(x=df[feature], y="Vehicles", data=df, hue="Junction", palette=colors)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

# Countplot of Traffic Over Years
plt.figure(figsize=(12, 5), facecolor="#627D78")
count = sns.countplot(data=df, x=df["Year"], hue="Junction", palette=colors)
count.set_title("Count Of Traffic On Junctions Over Years")
count.set_ylabel("Number of Vehicles")
count.set_xlabel("Date")
plt.show()

# Pivoting Data for Junctions
df_J = data.pivot(columns="Junction", index="DateTime")
df_1 = df_J[[('Vehicles', 1)]]
df_2 = df_J[[('Vehicles', 2)]]
df_3 = df_J[[('Vehicles', 3)]]
df_4 = df_J[[('Vehicles', 4)]]
df_4 = df_4.dropna()

# Dropping MultiIndex Level
for df in [df_1, df_2, df_3, df_4]:
    df.columns = df.columns.droplevel(level=1)

# Function to Plot Dataframes
def Sub_Plots4(df_1, df_2, df_3, df_4, title):
    fig, axes = plt.subplots(4, 1, figsize=(15, 8), facecolor="#627D78", sharey=True)
    fig.suptitle(title)
    sns.lineplot(ax=axes[0], data=df_1, color=colors[0]).set(ylabel="Junction 1")
    sns.lineplot(ax=axes[1], data=df_2, color=colors[1]).set(ylabel="Junction 2")
    sns.lineplot(ax=axes[2], data=df_3, color=colors[2]).set(ylabel="Junction 3")
    sns.lineplot(ax=axes[3], data=df_4, color=colors[3]).set(ylabel="Junction 4")
    plt.show()

# Plotting Before Transformation
Sub_Plots4(df_1, df_2, df_3, df_4, "Dataframes Before Transformation")

# Normalize Function
def Normalize(df, col):
    avg, std = df[col].mean(), df[col].std()
    df_normalized = (df[col] - avg) / std
    return df_normalized.to_frame(), avg, std

def Difference(df, col, interval):
    diff_values = [df[col][i] - df[col][i - interval] for i in range(interval, len(df))]
    # Pad with NaN to match the original length
    return [np.nan] * interval + diff_values

# Normalizing and Differencing
df_N1, _, _ = Normalize(df_1, "Vehicles")
df_N1["Diff"] = Difference(df_N1, "Vehicles", interval=168)  # Weekly difference
df_N1 = df_N1.dropna()

df_N2, _, _ = Normalize(df_2, "Vehicles")
df_N2["Diff"] = Difference(df_N2, "Vehicles", interval=24)  # Daily difference
df_N2 = df_N2.dropna()

df_N3, _, _ = Normalize(df_3, "Vehicles")
df_N3["Diff"] = Difference(df_N3, "Vehicles", interval=1)  # Hourly difference
df_N3 = df_N3.dropna()

df_N4, _, _ = Normalize(df_4, "Vehicles")
df_N4["Diff"] = Difference(df_N4, "Vehicles", interval=1)  # Hourly difference
df_N4 = df_N4.dropna()

# Splitting Dataset
def Split_data(df):
    training_size = int(len(df) * 0.90)
    train, test = df[0:training_size], df[training_size:]
    return train.values.reshape(-1, 1), test.values.reshape(-1, 1)

J1_train, J1_test = Split_data(df_N1["Diff"])
J2_train, J2_test = Split_data(df_N2["Diff"])
J3_train, J3_test = Split_data(df_N3["Diff"])
J4_train, J4_test = Split_data(df_N4["Diff"])

# Features and Targets for XGBoost
def Prepare_X_y(data):
    X, y = [], []
    steps = 32
    for i in range(steps, len(data)):
        X.append(data[i - steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_trainJ1, y_trainJ1 = Prepare_X_y(J1_train)
X_testJ1, y_testJ1 = Prepare_X_y(J1_test)

X_trainJ2, y_trainJ2 = Prepare_X_y(J2_train)
X_testJ2, y_testJ2 = Prepare_X_y(J2_test)

X_trainJ3, y_trainJ3 = Prepare_X_y(J3_train)
X_testJ3, y_testJ3 = Prepare_X_y(J3_test)

X_trainJ4, y_trainJ4 = Prepare_X_y(J4_train)
X_testJ4, y_testJ4 = Prepare_X_y(J4_test)

# Flatten for XGBoost
def FlattenForXGB(X, y):
    return X.reshape(X.shape[0], X.shape[1]), y

X_trainJ1_flat, y_trainJ1_flat = FlattenForXGB(X_trainJ1, y_trainJ1)
X_testJ1_flat, y_testJ1_flat = FlattenForXGB(X_testJ1, y_testJ1)

X_trainJ2_flat, y_trainJ2_flat = FlattenForXGB(X_trainJ2, y_trainJ2)
X_testJ2_flat, y_testJ2_flat = FlattenForXGB(X_testJ2, y_testJ2)

X_trainJ3_flat, y_trainJ3_flat = FlattenForXGB(X_trainJ3, y_trainJ3)
X_testJ3_flat, y_testJ3_flat = FlattenForXGB(X_testJ3, y_testJ3)

X_trainJ4_flat, y_trainJ4_flat = FlattenForXGB(X_trainJ4, y_trainJ4)
X_testJ4_flat, y_testJ4_flat = FlattenForXGB(X_testJ4, y_testJ4)

# XGBoost Model and Predictions
def XGB_model(X_train, y_train, X_test):
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test)

PredJ1 = XGB_model(X_trainJ1_flat, y_trainJ1_flat, X_testJ1_flat)
PredJ2 = XGB_model(X_trainJ2_flat, y_trainJ2_flat, X_testJ2_flat)
PredJ3 = XGB_model(X_trainJ3_flat, y_trainJ3_flat, X_testJ3_flat)
PredJ4 = XGB_model(X_trainJ4_flat, y_trainJ4_flat, X_testJ4_flat)

# RMSE Calculation
def RMSE_Value(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print(f"RMSE: {rmse:.2f}")
    return rmse

# Plot Predictions
def PredictionsPlot(test, predicted, junction_index):
    plt.figure(figsize=(12, 5), facecolor="#627D78")
    plt.plot(test, color=colors[junction_index], label="True Value", alpha=0.5)
    plt.plot(predicted, color="#627D78", label="Predicted Values")
    plt.title(f"Traffic Prediction Vs True Values (Junction {junction_index + 1})")
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Vehicles")
    plt.legend()
    plt.show()

# Evaluate Junctions
print("Junction 1 Results:")
RMSE_J1 = RMSE_Value(y_testJ1_flat, PredJ1)
PredictionsPlot(y_testJ1_flat, PredJ1, 0)

