#general
import io

# data
import numpy as np
import pandas as pd

# machine learning
import keras

# data visualization
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

#dataset load
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

# Updates dataframe to use specific columns.
training_df = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]

#print('Read dataset completed successfully.')
#print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
#training_df.head(200)

print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
training_df.describe(include='all')