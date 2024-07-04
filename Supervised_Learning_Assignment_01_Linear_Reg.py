# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1: Load the dataset using pandas
df = pd.read_csv("C:\\Users\\hp\\Desktop\\Data_Science_Course\\Course_02_Python\\06_Machine_Learning_using_Python\\Assignments\\Supervised_learning\\Assignment_01_Linear_regression\\data.csv")

sns.lineplot(data=df,x="YearsExperience",y="Salary")

# Step 2: Extract data from the 'years_experience' column into a variable named X
X = df['YearsExperience'].values.reshape(-1, 1)

# Step 3: Extract data from the 'salary' column into a variable named Y
Y = df['Salary'].values

# Step 4: Divide the dataset into two parts for training and testing in a 66% and 33% proportion
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Step 5: Create and train Linear Regression Model on the training set
model = LinearRegression()
model.fit(X_train, Y_train)

# Step 6: Make predictions based on the testing set using the trained model
Y_pred = model.predict(X_test)

# Step 7: Check the performance by calculating the r2 score of the model
r2_score_value = r2_score(Y_test, Y_pred)

# Print the R2 score
print("R2 Score:", r2_score_value)

