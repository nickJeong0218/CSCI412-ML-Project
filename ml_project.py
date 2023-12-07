# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score


# Define paths to the datasets
path1 = "CO2_Emissions_From_Energy_Consumption_Transportation.csv"
path2 = "Transportation_Data_-_Monthly_Data.csv"

# Function to load and preprocess the data
def openData(path):
  # Load the data
  df = pd.read_csv(path)
  title = path.split('.')[0]

  # Replace "Not Available" with NaN
  df.replace("Not Available", pd.NA, inplace=True)

  # Convert columns to numeric (if not already)
  df = df.apply(pd.to_numeric, errors='ignore')

  # Fill NaN values with 0
  df.fillna(0, inplace=True)
  print("Nulls replaced with 0")
  return df,title

# Function to visualize the data using scatter plots
def visulizeDataScatter(df):
  # Extract year from 'Month' column and convert it to numeric
  df['Year'] = pd.to_numeric(df['Month'].str.split().str[0])

  # Create subplots
  plt.figure(figsize=(50, 20))

  # Plot each feature against the year
  for i, feature in enumerate(df.columns[1:-1], 1):
    plt.subplot(3, 6, i)
    plt.scatter(x=df['Year'], y=df[feature])
    plt.xlabel('Year')
    plt.ylabel(feature)

  plt.show()

# Function to visualize the data using line plots
def visulizeDataTrendLine(df):
  # Extract year from 'Month' column and convert it to numeric
  df['Year'] = pd.to_numeric(df['Month'].str.split().str[0])

  # Calculate the mean for each year
  average_per_year = df.groupby('Year').mean(numeric_only=True)

  # Plot the average values for each year
  plt.figure(figsize=(15, 8))
  for feature in average_per_year.columns[:-5]:
      plt.plot(average_per_year.index, average_per_year[feature], label=feature)

  plt.xlabel('Year')
  plt.ylabel('BCU PER TRILLION')
  plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
  plt.grid(True)
  plt.show()

# Function to visualize the data using linear regression
def visulizeLinearRegression(df):
  # Create a new figure with a specified size
  fig = plt.figure(figsize=(50,30))

  # Loop over each feature in the dataframe, excluding the first column (Year)
  for i, feature in enumerate(df.columns[1:-1], 1):
    # Set the independent variable (X) as the Year and the dependent variable (y) as the current feature
    X = df[["Year"]]
    y = df[[feature]]

    # Create a linear regression model
    lin_model = LinearRegression()
    # Fit the model to the data
    lin_model.fit(X, y)

    # Add a subplot to the figure
    ax = fig.add_subplot(3,5,i)

    # Plot the original data as gray dots
    ax.scatter(X, y, color='gray')
    # Plot the model's predictions as a red line
    ax.plot(X, lin_model.predict(X),color='red')
    # Set the title of the subplot to the current feature
    ax.set_title(f'{feature} on Year')
    # Label the x-axis as Year
    ax.set_xlabel('Year')
    # Label the y-axis as the current feature
    ax.set_ylabel(feature)

  # Display the figure
  plt.show()


# Function to visualize the data using polynomial features
def visulizePolynomialFeatures(df):
  # Create a new figure with a specified size
  fig = plt.figure(figsize=(50,30))

  # Loop over each feature in the dataframe, excluding the first column (Year)
  for i, feature in enumerate(df.columns[1:-1], 1):
    # Set the independent variable (X) as the Year and the dependent variable (y) as the current feature
    X = df[["Year"]]
    y = df[[feature]]

    # Create a PolynomialFeatures transformer with degree 2
    poly = PolynomialFeatures(degree=2)
    # Transform the independent variable using the polynomial features transformer
    X_poly = poly.fit_transform(X)

    # Create a linear regression model
    poly_model = LinearRegression()
    # Fit the model to the transformed data
    poly_model.fit(X_poly, y)

    # Add a subplot to the figure
    ax = fig.add_subplot(3,5,i)

    # Plot the original data as gray dots
    ax.scatter(X, y, color='gray')
    # Plot the model's predictions as a red line
    ax.plot(X, poly_model.predict(X_poly),color='red')
    # Set the title of the subplot to the current feature
    ax.set_title(f'{feature} on Year')
    # Label the x-axis as Year
    ax.set_xlabel('Year')
    # Label the y-axis as the current feature
    ax.set_ylabel(feature)

  # Display the figure
  plt.show()


# Function to visualize the data using spline transformation
def visulizeSplineTransformation(df):
  # Create a new figure with a specified size
  fig = plt.figure(figsize=(50,30))

  # Loop over each feature in the dataframe, excluding the first column (Year)
  for i, feature in enumerate(df.columns[1:-1], 1):
    # Set the independent variable (X) as the Year and the dependent variable (y) as the current feature
    X = df[["Year"]]
    y = df[[feature]]

    # Create a SplineTransformer with 5 knots
    spline = SplineTransformer(n_knots=5)
    # Transform the independent variable using the spline transformer
    X_spline = spline.fit_transform(X)

    # Create a linear regression model
    spline_model = LinearRegression()
    # Fit the model to the transformed data
    spline_model.fit(X_spline, y)

    # Add a subplot to the figure
    ax = fig.add_subplot(3,5,i)

    # Plot the original data as gray dots
    ax.scatter(X, y, color='gray')
    # Plot the model's predictions as a red line
    ax.plot(X, spline_model.predict(X_spline),color='red')
    # Set the title of the subplot to the current feature
    ax.set_title(f'{feature} on Year')
    # Label the x-axis as Year
    ax.set_xlabel('Year')
    # Label the y-axis as the current feature
    ax.set_ylabel(feature)

  # Display the figure
  plt.show()


# Function to aggregate two dataframes
def aggregate(df1,df2):
  # Concatenate the two dataframes along the columns
  df_agg = pd.concat([df1,df2],axis=1)

  # Convert all columns to numeric, ignoring errors
  df_agg = df_agg.apply(pd.to_numeric, errors='ignore')

  # Select only the necessary columns
  df_agg = df_agg[['Coal','Natural Gas','Petroleum','Biomass','End-Use Energy Consumed',
                 'Coal Transportation Sector CO2 Emissions','Natural Gas Transportation Sector CO2 Emissions',
                 'Petroleum, Excluding Biofuels, Transportation Sector CO2 Emissions', 'Biomass Transportation Sector CO2 Emissions',
                 'Transportation Share of Electric Power Sector CO2 Emissions']]

  # Display the aggregated dataframe
  print(df_agg.head())

  return df_agg

# Function to plot fuel consumption against CO2 emissions
def fuelVsConsump(df_agg):
  # Create a figure
  fig = plt.figure(figsize=(15,10))

  # Get the halfway point of the columns
  half = len(df_agg.columns)//2


  # For each pair of fuel and emission columns
  for i,(fuel, emission) in enumerate(zip(df_agg.columns[:half],df_agg.columns[half:]),1):
    # Extract the data for the current pair
    X = df_agg[[fuel]]
    y = df_agg[[emission]]

    # Create a subplot
    ax = fig.add_subplot(2,3,i)

    # Plot the data
    ax.scatter(X,y)
    ax.set_xlabel(fuel)
    ax.set_ylabel(emission)

  # Show the plot
  plt.show()

# Function to plot linear regression models for fuel consumption against CO2 emissions
def linearFuelVsConsump(df_agg):
  # Create a figure
  fig = plt.figure(figsize=(15,10))

  # Get the halfway point of the columns
  half = len(df_agg.columns)//2

  # For each pair of fuel and emission columns
  for i,(fuel, emission) in enumerate(zip(df_agg.columns[:half],df_agg.columns[half:]),1):
    # Extract the data for the current pair
    X = df_agg[[fuel]]
    y = df_agg[[emission]]

    # Fit a linear regression model to the data
    lin_model = LinearRegression()
    lin_model.fit(X, y)

    # Create a subplot
    ax = fig.add_subplot(2,3,i)

    # Plot the data and the model's predictions
    ax.scatter(X,y,color='gray')
    ax.plot(X,lin_model.predict(X))
    ax.set_xlabel(fuel)
    ax.set_ylabel(emission)

    # Print the model's coefficient and intercept
    print(f'{fuel} has {np.ravel(lin_model.coef_)[0]:.5f} ratio of emissions (Mmt/Btu)')

  # Show the plot
  plt.show()

# Function to plot the predictions of a spline transformation
def predictionSplineTransformation(df):
    # Create a new figure with a specified size
    fig = plt.figure(figsize=(50,30))

    # Loop over each feature in the dataframe, excluding the first column (Year)
    for i, feature in enumerate(df.columns[1:-1], 1):
        # Create a KFold object
        kf = KFold(n_splits=5)

        # Set the independent variable (X) as the Year and the dependent variable (y) as the current feature
        X = df[["Year"]]
        y = df[[feature]]

        # Create a SplineTransformer with knots
        spline = SplineTransformer(n_knots=3)

        # Create a linear regression model
        spline_model = LinearRegression()

        # Add a subplot to the figure
        ax = fig.add_subplot(3,5,i)

        # Plot the original data as gray dots
        ax.scatter(X, y, color='gray')

        X_spline = spline.fit_transform(X)

        # Initialize lists to store error metrics for each fold
        mae_list = []
        mse_list = []
        r2_list = []
        cv_scores = []

        # For each split, fit the model to the training data and evaluate it on the test data
        for train_index, test_index in kf.split(X_spline):
            X_train, X_test = X_spline[train_index], X_spline[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Fit the model to the transformed data
            spline_model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = spline_model.predict(X_test)

            # Calculate error metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            cv_score = cross_val_score(spline_model, X_spline, y, cv=kf)

            # Store the error metrics for this fold
            mae_list.append(mae)
            mse_list.append(mse)
            r2_list.append(r2)
            cv_scores.append(cv_score)

        # Calculate the average error metrics
        avg_mae = np.mean(mae_list)
        avg_mse = np.mean(mse_list)
        avg_r2 = np.mean(r2_list)
        avg_cv_score = np.mean(cv_scores)

        # Print the MAE, MSE, R2 Score, and cross-validation score
        print(f'Feature: {feature}, MAE: {avg_mae}, MSE: {avg_mse}, R2 Score: {avg_r2}, Cross-Validation Score: {avg_cv_score}')


        # Plot the model's predictions as red dots
        ax.scatter(np.mean(X_test,axis=1).reshape(-1,1), y_pred, color='red')

        # Set the title of the subplot to the current feature
        ax.set_title(f'{feature} on Year')

        # Label the x-axis as Year
        ax.set_xlabel('Year')

        # Label the y-axis as the current feature
        ax.set_ylabel(feature)

    # Display the figure
    plt.show()




def mainCall():
  # Load and preprocess the data  
  print("Dataset 1 opening")#Use CO2_Emissions_From_Energy_Consumption_Transportation
  data1, title1 = openData(path1)
  print("Loaded dataset ", title1, end="\n")

  print("Dataset 1 opening")#Use Transportation_Data_-_Monthly_Data
  data2, title2 = openData(path2)
  print("Loaded dataset ", title2, end="\n")

  # Visulize the dataset CO2_Emissions_From_Energy_Consumption_Transportation
  print("\n\n\n\n")
  print("Visulizations from", title1)

  print("\nvisulize Data Scatter plot")
  visulizeDataScatter(data1)
  print("Observations: There are a variety of observations that can be made from this scatter plot, we can see all of the different CO2 emissions based on the consumptions \n",
        "of specific types of fuels. We can make specific observations within our data such as things like: Coal is not being used as much in today's age. And that pandemic \n",
        "data makes predictions harder due to its tendency to be perceived as outlier data.")
  
  print("\nvisulize Data Trend Line")
  visulizeDataTrendLine(data1)
  print("Observations: Here we can see the pollution directly compared to the fuel types, we can see that distillate fuel oil otherwise known as petroleum is the most \n",
        "prominent polluter  within the fuel types, the next being jet fuel which hovers at around half the pollution from distillate fuel oil. \n",
        "And the third notable polluter is Natural gas making a measly 4-6 BTU per trillion.")
  
  print("\nvisulize Linear Regression")
  visulizeLinearRegression(data1)
  print("Observations: Within this cluster of graphs we can see the various regression lines, these lines tend to show where the data is headed\n",
        "but due to its non-confirm nature it leads to underfitting witin the data. So this is a great visulization for the data but provides no indepth analysis for us.\n",
        "Another issue is that it start \nto go negative for coal predictions.")
  
  print("\nvisulize Polynomial Features")
  visulizePolynomialFeatures(data1)
  print("Observations: Within this cluster of graphs we see a much better fit for the data, the lines flow with the data and offer great predictions for data prior to 2019.\n", 
        "The main issue with this data and the polynomial prediction is that it isnt as precise as we’ed like. This lack of precision makes most values predicted after \n",
        "2020-2021 predict in a negative manner as there is not enough pandemic “recovey” data to correct the curve. This trend incorrectly predicts values after these dates. ")
  
  print("\nvisulize Spline Transformer")
  visulizeSplineTransformation(data1)
  print("Observations: Within this cluster of graphs we see spline transformer at work. As you can see the prediction curve follows the data much more accurately.\n",
        "The curve follows \ndata well but isnt overfitting in nature. The issue with the pandemic data is still prevalent but in some graphs there are upward slopes.\n",
        "This is a step \nin the correct direction, but there is still an need to fix certain data clusters.")

  # Visulize the dataset Transportation_Data_-_Monthly_Data
  print("\n\n\n\n")
  print("Visulizations are from", title2)

  print("\nvisulize Data Scatter Plot")
  visulizeDataScatter(data2)
  print("Observations: Within these cluster charts, we can see all the consomptions from the transportation sector the key difference from the charts above is \n",
        "that they where Pollution outputs, whilst this dataset is of consumption. As we can see the previous dataset closely follows this dataset\n",
        "as pollution and consumption are closley related. And we also see the same downfalls and shortcomings of the pandemic data.")
  
  print("\nvisulize Data Trend Line")
  visulizeDataTrendLine(data2)
  print("Observations: This line graph compares the consumptions of the different fuel types and shows a strong correlation between total fossil fuels and the consumption of petroleum.\n",
        "Another surprising feature within the data is the lack of impact there was on natural gas and Biomass during the pandemic.",
        "For petroleum we see a steep decline in consumption showing lows as far as consumption from 30 years ago but both biomass and natural gas remain unaffected. ")
  
  print("\nvisulize Linear Regression")
  visulizeLinearRegression(data2)
  print("Observations: Just as in the pollution this shows the general slopes of the graphs but doesn't do a good job at predicting exact values this causes the model to be severely underfit.\n",
        "And also as with the other data it predicts negatives for coal.")
  
  print("\nvisulize Polynomial Features")
  visulizePolynomialFeatures(data2)
  print("Observations: Within this cluster of graphs, we see a much better fit for the data, the lines flow with the data and offer much better predictions than with the linear model.\n",
        "But asnobserved previously we can see that this is still not good enough as the pandemic data severely destroys the estimated future slopes.")
  
  print("\nvisulize Spline Transformer")
  visulizeSplineTransformation(data2)
  print("Observations: Within this cluster of graphs, yet again see spline transformer at work. Just as with the other dataset this conforms to the data much better\n",
        "and offers much more valid predictions compared to the other results we can see the pandemic data still affects the charts but that the model does not overly focus on pandemic data. ")

  print("\n\n\n\n")
  print("\nAggregate the Data Sets", title1, "and", title2, "into one dataframe")
  df_agg = aggregate(data1,data2)

  print("Visulize the Aggregated Data")
  fuelVsConsump(df_agg)

  linearFuelVsConsump(df_agg)


  # predictionSplineTransformation(data2)
  # print("Observations: ")

  # predictionSplineTransformation(data1)
  # print("Observations: ")

  # predictionSplineTransformation(df_agg)
  # print("Observations: ")




mainCall()