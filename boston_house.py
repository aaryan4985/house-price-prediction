import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Import the necessary dataset
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
california = fetch_california_housing()


data = pd.DataFrame(california.data, columns=california.feature_names)


data['Price'] = california.target

print(data.head())


data['RoomOccupancy'] = data['AveRooms'] / data['AveOccup']


X = data.drop('Price', axis=1)
y = data['Price']

