# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv") 

# Step 3: Explore the data

# print("Data Head",data.head())
# print("Data Info",data.info())
# print("Data Describe", data.describe())

# Step 4: Visualize the relationship between total_bill and tip of original data
plt.scatter(data['total_bill'], data['tip'])
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.title('Total Bill vs Tip of Original Data')
plt.show()


#Step 4.1

# Check for missing values
# print("Missing values in each column:")
# print(data.isnull().sum()) 
# As there is no missing valuea so no need to fix missing values!

# Step 4.2
# Select numerical columns only (e.g., 'total_bill', 'tip')


numerical_cols = ['total_bill', 'tip', 'size']

# Step 4.3
# Detect outliers based on the IQR method
for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))]
    print(f"Outliers in {col}:", outliers[col].values)

# Step 5: Split the data into training and testing sets
X = data[['total_bill']]
y = data['tip']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Linear Regression model
model1 = LinearRegression()
model1.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model1.predict(X_test)



# Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")



#Step 9
# Filter data to remove outliers
cleaned_data = data.copy()

for col in numerical_cols:
    Q1 = cleaned_data[col].quantile(0.25)
    Q3 = cleaned_data[col].quantile(0.75)
    IQR = Q3 - Q1
    # Retain rows that are not outliers
    cleaned_data = cleaned_data[(cleaned_data[col] >= (Q1 - 1.5 * IQR)) & (cleaned_data[col] <= (Q3 + 1.5 * IQR))]

# Step 4 w.r.t cleaned data: Visualize the relationship between total_bill and tip of cleaned data
plt.scatter(cleaned_data['total_bill'], cleaned_data['tip'])
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.title('Total Bill vs Tip of Cleaned Data')
plt.show()

# print("Data before Outlier Removal: ", data)
# print("Data after outlier removal:", cleaned_data)

# the original data has 244 rows and after cleaning 22 rows have been removed and get cleaned data 

# Step 5 w.r.t cleaned data
# Plotting the regression line
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.legend()
plt.show()


# Step 6 w.r.t cleaned data: Split the data into training and testing sets
X2 = cleaned_data[['total_bill']]
y2 = cleaned_data['tip']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Step 7 w.r.t cleaned data: Train the Linear Regression model with CLeaned Data!
model2 = LinearRegression()
model2.fit(X_train2, y_train2)

# Step 8 w.r.t cleaned data: Make predictions
y_pred2 = model2.predict(X_test2)



# Step 9 w.r.t cleaned data: Evaluate the model 2
mse_model2 = mean_squared_error(y_test2, y_pred2)
r2_model2 = r2_score(y_test2, y_pred2)

print(f"Mean Squared Error of Model with Cleaned Data: {mse_model2}")
print(f"R-squared Error of Model with Original Data: {r2_model2}")


# Step 9 w.r.t cleaned data: Plotting the regression line for cleaned Data
plt.scatter(X_test2, y_test2, color='blue', label='Actual')
plt.plot(X_test2, y_pred2, color='red', label='Regression Line')
plt.title("Total Bill Vs Tip of Cleaned Data")
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.legend()
plt.show()