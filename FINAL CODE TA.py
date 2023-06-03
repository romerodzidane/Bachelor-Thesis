#Developed by: Romero Deandhito Zidane - 12219090
#Made for the purposes of bachelor thesis
#The growth curve models that are used in this predictive model are the growth curve models proposed by Liu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

df = pd.read_excel(r'C:\Users\HP\Downloads\Documents\TA\Model\Running.xlsx', sheet_name='Peripheral')

# Convert the date column to datetime type if it's not already in datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Define the start and end dates for the desired time range (YY-MM-DD)
start_date = pd.to_datetime("2014-01-01")
end_date = pd.to_datetime("2019-01-01")
Year = 5

# Filter the data within the specified time range
filtered_df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

# Access the specific columns of interest from the filtered data
selected_data = filtered_df[["Month", "Np (MMSTB)", "Monthly Np (MMSTB)", "Water Cut"]]

# Print the extracted data
print(selected_data)

# Filter the data within the specified time range
forecasted_df = df[(df["Date"] >= end_date)]

WC_forecast = []
Np_forecast = []
month = []
galat_WC = []
galat_Np = []

# Define the equation
def equation_fw(t, a1, b1, c1, d1):
    return (1 - (d1+1) / ((np.exp(a1*t**b1)) + d1))**c1

# Define your data
fw = filtered_df["Water Cut"]
t = filtered_df["Month"]

# Define the initial guesses for the constants
initial_guess1 = [1, 1, 1, 1]  # You can modify these values as per your requirement

# Set the maximum number of function evaluations
maxfev1 = 4000

constants1 = None

# Define the optimization methods in the desired order
methods1 = ['lm', 'trf', 'dogbox']

for method in methods1:
    try:
        # Perform the curve fitting
        constants1, _ = curve_fit(equation_fw, t, fw, p0=initial_guess1, method=method, maxfev=maxfev1)
        a1_fit, b1_fit, c1_fit, d1_fit = constants1

        print(f"Optimization success with method '{method}'.")

        # Print the fitted constants
        print("Fitted Constants:")
        print("a =", a1_fit)
        print("b =", b1_fit)
        print("c =", c1_fit)
        print("d =", d1_fit)

        # Plot the equation curve
        t_range = np.linspace(t.min(), t.max(), 100)
        fw_result = equation_fw(t_range, a1_fit, b1_fit, c1_fit, d1_fit)
        plt.plot(t_range, fw_result, label='Equation Water Cut')

        # Plot the data points
        plt.scatter(t, fw, label='Data')

        # Set plot title and labels
        plt.title('Water Cut vs Time')
        plt.xlabel('Time')
        plt.ylabel('Water Cut')

        # Add legend
        plt.legend()

        # Display the plot
        plt.show()

        for x in range(0, len(forecasted_df)):
            time = forecasted_df["Month"].iloc[x]
            WC_f = equation_fw(time, a1_fit, b1_fit, c1_fit, d1_fit)
            error_WC = np.abs((forecasted_df['Water Cut'].iloc[x] - WC_f) * 100 / forecasted_df['Water Cut'].iloc[x])
            WC_forecast.append(WC_f)
            month.append(time)
            galat_WC.append(error_WC)

        break  # Exit the loop if successful

    except RuntimeError:
        print(f"Optimization failed with method '{method}'. Trying the next method.")

if constants1 is None:
    print("All optimization methods failed.")

# Define the equation
def equation_Np(t, Nrmax, a2, b2, c2, d2):
    return Nrmax * ((np.exp(a2 * t**b2) - 1) / (np.exp(a2 * t**b2) + d2))**c2

# Define your data
Np = filtered_df["Np (MMSTB)"]
OOIP = 13.4
Np_last = 7

# Define the initial guesses for the constants
initial_guess2 = [Np_last, 1, 1, 1, 1]  # You can modify these values as per your requirement

# Define the bounds
bounds = ([Np_last, 0, 0, 0, 0], [OOIP, 100, 10, np.inf, np.inf])

# Define the optimization methods in the desired order
methods2 = ['trf', 'dogbox']

# Set the maximum number of function evaluations
maxfev2 = 4000

constants2 = None

for method in methods2:
    try:
        # Perform the curve fitting
        constants2, _ = curve_fit(equation_Np, t, Np, p0=initial_guess2, bounds=bounds, method=method, maxfev=maxfev2)
        Nrmax_fit, a2_fit, b2_fit, c2_fit, d2_fit = constants2

        print(f"Optimization success with method '{method}'.")

        # Print the fitted constants
        print("Fitted Constants:")
        print("a =", a2_fit)
        print("b =", b2_fit)
        print("c =", c2_fit)
        print("d =", d2_fit)
        print("Estimated Nrmax:", Nrmax_fit)

        # Plot the equation curve
        Np_result = equation_Np(t, Nrmax_fit, a2_fit, b2_fit, c2_fit, d2_fit)
        plt.plot(t, Np_result, label='Equation Np')

        # Plot the data points
        plt.scatter(t, Np, label='Data')

        # Set plot title and labels
        plt.title('Np vs Time')
        plt.xlabel('Time')
        plt.ylabel('Np (MMSTB)')

        # Add legend
        plt.legend()

        # Display the plot
        plt.show()

        qo_forecast = []

        for i in range(0, len(forecasted_df)):
            time = forecasted_df["Month"].iloc[i]
            Np_f = equation_Np(time, Nrmax_fit, a2_fit, b2_fit, c2_fit, d2_fit)
            error_Np = np.abs((forecasted_df['Np (MMSTB)'].iloc[i] - Np_f) * 100 / forecasted_df['Np (MMSTB)'].iloc[i])
            Np_forecast.append(Np_f)
            galat_Np.append(error_Np)
            if i == 0:
                day = (filtered_df["Date"].iloc[-1] - filtered_df["Date"].iloc[-2]).days
                qo_0 = (forecasted_df["Np (MMSTB)"].iloc[i] - filtered_df["Np (MMSTB)"].iloc[-2])*1000000/day
                qo_forecast.append(qo_0)
            else:
                day = (forecasted_df["Date"].iloc[i] - forecasted_df["Date"].iloc[i-1]).days
                qo_f = (Np_forecast[i] - Np_forecast[i-1])*1000000/day
                qo_forecast.append(qo_f)

        break  # Exit the loop if successful

    except RuntimeError:
        print(f"Optimization failed with method '{method}'. Trying the next method.")

if constants2 is None:
    print("All optimization methods failed.")

# Create a new DataFrame to store the forecasted WC and t values
forecast = {'Date':forecasted_df['Date'],'Month':month, 'Np (MMSTB)':Np_forecast, 'qo (STBD)':qo_forecast, 'WC':WC_forecast, 'Galat Np':galat_Np, 'Galat Water Cut':galat_WC}
forecast_df = pd.DataFrame(forecast)

# Print the forecasted Np and qo values
print("Forecast Result:")
print(forecast_df)

# Assuming water cut actual columns in a DataFrame called df
WC_actual = forecasted_df['Water Cut']
Np_actual = df['Np (MMSTB)']
Np_predicted = []

for y in range(0, len(df)):
    time = df["Month"].iloc[y]
    Np_pred = equation_Np(time, Nrmax_fit, a2_fit, b2_fit, c2_fit, d2_fit)
    Np_predicted.append(Np_pred)

# Convert forecast_df["Date"] to numeric values
date_numeric = mdates.date2num(filtered_df["Date"])

# Generate the time range for the equation
t1_range = np.linspace(min(date_numeric), max(date_numeric), 100)

# Plot the equation curve for water cut
fw_result = equation_fw(t_range, a1_fit, b1_fit, c1_fit, d1_fit)
plt.plot(t1_range, fw_result, color='green', label='Growth Curve Water Cut')
plt.plot(forecast_df["Date"], WC_forecast, color='red', label='Forecast Data')

# Plot the actual data for water cut
plt.scatter(df["Date"], df["Water Cut"], color='blue', label='Actual Data')

plt.title('Water Cut vs Time')
plt.xlabel('Time')
plt.ylabel('Water Cut')

plt.legend()
plt.show()

# Plot the equation curve for water cut
Np_result = equation_Np(t_range, Nrmax_fit, a2_fit, b2_fit, c2_fit, d2_fit)
plt.plot(t1_range, Np_result, color='green', label='Growth Curve Np')
plt.plot(forecast_df["Date"], Np_forecast, color='red', label='Forecast Data')

# Plot the actual data for water cut
plt.scatter(df["Date"], df["Np (MMSTB)"], color='blue', label='Actual Data')

plt.title('Np vs Time')
plt.xlabel('Time')
plt.ylabel('Np (MMSTB)')

plt.legend()
plt.show()

#Find the minimum and maximum percent error for water cut
maximum_galWC = np.max(galat_WC)
minimum_galWC = np.min(galat_WC)

#Print the minimum and maximum percent error for water cut
print("Maximum percent error WC:", maximum_galWC)
print("Minimum percent error WC:", minimum_galWC)

#Find the minimum and maximum percent error for water cut
maximum_galNp = np.max(galat_Np)
minimum_galNp = np.min(galat_Np)

#Print the minimum and maximum percent error for water cut
print("Maximum percent error Np:", maximum_galNp)
print("Minimum percent error Np:", minimum_galNp)

# Calculate the RMSE
rmse_WC = np.sqrt(mean_squared_error(WC_actual, WC_forecast))

# Calculate R-squared value
r2_Np = r2_score(Np_actual, Np_predicted)

# Print the RMSE value
print("RMSE Water Cut:", rmse_WC)

# Print the R-squared value
print("R-squared Np:", r2_Np)

# Specify the existing file path
file_path = 'C:/Users/HP/Downloads/Documents/TA/Model/Forecast Results.xlsx'

# Load the existing workbook
wb = openpyxl.load_workbook(file_path)

# Create a new sheet and set its name
sheet_name = (f"Liu {Year}y")
wb.create_sheet(title=sheet_name)

# Get the new sheet
new_sheet = wb[sheet_name]

# Write the forecasted data to the new sheet
for row in dataframe_to_rows(forecast_df, index=False, header=True):
    new_sheet.append(row)

# Save the changes
wb.save(file_path)

print("Results added to a new sheet in:", file_path)