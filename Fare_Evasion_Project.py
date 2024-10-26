# Arnav Sastry
# Fare Evasion Project for 2024 MTA Open Data Challenge

import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import webbrowser
import os
import pingouin as pg
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

def open_pdf(filepath):
    absolute_path = os.path.abspath(filepath)
    webbrowser.open(f"file://{absolute_path}")

pdf = PdfPages("Fare_Evasion_Project.pdf")

# Reading Data
bus_fares = pd.read_csv("../MTA_Project_Arnav_Sastry/MTA_Bus_Fare_Evasion__Beginning_2019_20241019.csv")
subway_fares = pd.read_csv("../MTA_Project_Arnav_Sastry/MTA_NYCT_Subway_Fare_Evasion__Beginning_2018_20241019.csv")

# Data Cleaning and Sorting
bus_to_drop = ["Ridership"]
bus_fares.drop(columns = bus_to_drop, inplace = True)
bus_fares.sort_values(["Trip Type"])
subway_fares.drop([0, 1, 2, 3], inplace = True)  
bus_fares_new = bus_fares.pivot_table(
    index = "Time Period", 
    columns = "Trip Type", 
    values = "Fare Evasion",
    aggfunc = "sum"
)
bus_fares_new = bus_fares_new.reset_index()
bus_fares_new.replace(0, np.nan, inplace = True)

# Exploratory Data Analysis

# Line Graph of Buses
plt.figure(figsize = (14, 6))
sns.lineplot(data = bus_fares_new, x = "Time Period", y = "Local", label = "Local", marker = "o")
sns.lineplot(data = bus_fares_new, x = "Time Period", y = "Express", label = "Express", marker = "o")
sns.lineplot(data = bus_fares_new, x = "Time Period", y = "SBS", label = "SBS", marker = "o")
plt.title("Bus Fare Evasion Between 2019 and 2024")
plt.xlabel("Time Period (Year-Quarter)")  
plt.ylabel("Fare Evasion (%)")
plt.xticks(ticks = range(len(bus_fares_new["Time Period"])), labels = bus_fares_new["Time Period"], rotation = 45)
pdf.savefig()
plt.close()

# Bar Graph of Buses
plt.figure(figsize = (14, 6))
sns.barplot(data = bus_fares, x = "Time Period", y = "Fare Evasion", hue = "Trip Type")
plt.title("Bus Fare Evasion Between 2019 and 2024")
plt.xlabel("Time Period (Year-Quarter)")  
plt.ylabel("Fare Evasion (%)")
plt.xticks(ticks = range(len(bus_fares_new["Time Period"])), labels = bus_fares_new["Time Period"], rotation = 45)
pdf.savefig()
plt.close()

# Line Graph of Subways
plt.figure(figsize = (14,6 ))
sns.lineplot(data = subway_fares, x = "Time Period", y = "Fare Evasion", marker = "o")
plt.title("Subway Fare Evasion Between 2019 and 2024")
plt.xlabel("Time Period (Year-Quarter)")  
plt.ylabel("Fare Evasion (%)")
plt.xticks(ticks = range(len(bus_fares_new["Time Period"])), labels = bus_fares_new["Time Period"], rotation = 45)
pdf.savefig()
plt.close()

# Bar Graph of Subways
plt.figure(figsize = (14, 6))
sns.barplot(data = subway_fares, x = "Time Period", y = "Fare Evasion")
plt.title("Subway Fare Evasion Between 2019 and 2024")
plt.xlabel("Time Period (Year-Quarter)")  
plt.ylabel("Fare Evasion (%)")
plt.xticks(ticks = range(len(bus_fares_new["Time Period"])), labels = bus_fares_new["Time Period"], rotation = 45)
pdf.savefig()
plt.close()

# Line Graph of Everything
plt.figure(figsize = (14, 6))
sns.lineplot(data = bus_fares_new, x = "Time Period", y = "Local", label = "Local Bus", marker = "o")
sns.lineplot(data = bus_fares_new, x = "Time Period", y = "Express", label = "Express Bus", marker = "o")
sns.lineplot(data = bus_fares_new, x = "Time Period", y = "SBS", label = "SBS Bus", marker = "o")
sns.lineplot(data = subway_fares, x = "Time Period", y = "Fare Evasion", label = "Subway", marker = "o")
plt.title("All Fare Evasion Between 2019 and 2024")
plt.xlabel("Time Period (Year-Quarter)")  
plt.ylabel("Fare Evasion (%)")
plt.xticks(ticks = range(len(bus_fares_new["Time Period"])), labels = bus_fares_new["Time Period"], rotation = 45)
pdf.savefig()
plt.close()

# Start of Linear Regressions
data_of_both = pd.merge(
    bus_fares_new[["Time Period", "Local", "Express", "SBS"]], 
    subway_fares[["Time Period", "Fare Evasion"]], 
    on = "Time Period", 
    how = "inner"
)
data_of_both.rename(columns = {"Fare Evasion": "Subway Evasion", "Local": "Local Bus Evasion", "Express": "Express Bus Evasion", "SBS": "SBS Bus Evasion"}, inplace = True)

# Local Bus vs. Subway
plt.figure(figsize = (14, 6))
sns.regplot(data = data_of_both, x = "Subway Evasion", y = "Local Bus Evasion")
plt.title("Local Bus Fare Evasion vs. Subway Fare Evasion")
plt.xlabel("Subway Fare Evasion (%)")  
plt.ylabel("Local Bus Fare Evasion (%)")
pdf.savefig()
plt.close()

# Express Bus vs. Subway
plt.figure(figsize = (14, 6))
sns.regplot(data = data_of_both, x = "Subway Evasion", y = "Express Bus Evasion")
plt.title("Express Bus Fare Evasion vs. Subway Fare Evasion")
plt.xlabel("Subway Fare Evasion (%)")  
plt.ylabel("Express Bus Fare Evasion (%)")
pdf.savefig()
plt.close()

# SBS Bus vs. Subway
plt.figure(figsize = (14, 6))
sns.regplot(data = data_of_both, x = "Subway Evasion", y = "SBS Bus Evasion")
plt.title("SBS Fare Evasion vs. Subway Fare Evasion")
plt.xlabel("Subway Fare Evasion (%)")  
plt.ylabel("SBS Fare Evasion (%)")
pdf.savefig()
plt.close()

# Finding Correlation Coefficients Between Subway and Bus Fare Evasions
print("Correlation Coefficients Between Subway Fare Evasions and Bus Fare Evasions")
corr_local = round(data_of_both["Subway Evasion"].corr(data_of_both["Local Bus Evasion"]), 4)
print("Correlation Coefficient Between Subway Fare Evasion and Local Bus Fare Evasion: r = " + str(corr_local))
corr_express = round(data_of_both["Subway Evasion"].corr(data_of_both["Express Bus Evasion"]), 4)
print("Correlation Coefficient Between Subway Fare Evasion and Express Bus Fare Evasion: r = " + str(corr_express))
corr_sbs = round(data_of_both["Subway Evasion"].corr(data_of_both["SBS Bus Evasion"]), 4)
print("Correlation Coefficient Between Subway Fare Evasion and SBS Fare Evasion: r = " + str(corr_sbs))
print()

# ANOVA to Obtain p-values
print("ANOVA Results for Subway Fare Evasions and Bus Fare Evasions")
print("Subway Fare Evasion vs. Local Bus Fare Evasion")
local_anova = pg.anova(data = data_of_both, dv = "Local Bus Evasion", between = "Subway Evasion", detailed = True)
print(local_anova)
print("Subway Fare Evasion vs. Express Bus Fare Evasion")
express_anova = pg.anova(data = data_of_both, dv = "Express Bus Evasion", between = "Subway Evasion", detailed = True)
print(express_anova)
print("Subway Fare Evasion vs. SBS Fare Evasion")
sbs_anova = pg.anova(data = data_of_both, dv = "SBS Bus Evasion", between = "Subway Evasion", detailed = True)
print(sbs_anova)
print()

# Predictive Modeling for Local Buses

# Building Decision Tree Model
empty = data_of_both[data_of_both["Local Bus Evasion"].isna()]["Time Period"]
data_of_both_new = data_of_both[~data_of_both["Time Period"].isin(empty)]
y = data_of_both_new["Local Bus Evasion"]
X = data_of_both_new[["Subway Evasion"]]
fare_evasion_model = DecisionTreeRegressor(random_state=1)
fare_evasion_model.fit(X, y)
predicted_bus_evasion = fare_evasion_model.predict(X)

# Validating Model
mean_abs_error = mean_absolute_error(y, predicted_bus_evasion)
print("For the Decision Tree Model:")
print("The mean absolute error of the in-sample data is: " + str(mean_abs_error))
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
fare_evasion_model = DecisionTreeRegressor()
fare_evasion_model.fit(train_X, train_y)
val_predictions = fare_evasion_model.predict(val_X)
out_mean_abs_error = mean_absolute_error(val_y, val_predictions)
print("The mean absolute error of the out-of-sample data is: " + str(out_mean_abs_error))
print()

# Visualizing Decision Tree Model
fig = plt.figure(figsize=(14, 6))  
plot_tree(fare_evasion_model, feature_names = ["Subway Evasion"], filled = True)
plt.title("Decision Tree for Local Bus Fare Evasion (%) Predicted by Subway Fare Evasion (%)")
pdf.savefig()
plt.close()

# Building Random Forest Model
fare_evasion_forest = RandomForestRegressor(random_state=1)
fare_evasion_forest.fit(train_X, train_y)
predicted_bus_forest = fare_evasion_forest.predict(val_X)

# Validating Model
mean_abs_error_forest = mean_absolute_error(val_y, predicted_bus_forest)
print("For the Random Forest Model:")
print("The mean absolute error is: " + str(mean_abs_error_forest))

# Visualizing Decision Tree Model
fig = plt.figure(figsize=(14, 6))  
tree = fare_evasion_forest.estimators_[0]
plot_tree(tree, feature_names = ["Subway Evasion"], filled = True)
plt.title("Random Forest for Local Bus Fare Evasion (%) Predicted by Subway Fare Evasion (%)")
pdf.savefig()
plt.close()

pdf.close()
open_pdf("Fare_Evasion_Project.pdf")