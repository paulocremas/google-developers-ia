#general
import io

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
import matplotlib.pyplot as plt

#dataset load
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

# Updates dataframe to use specific columns.
training_df = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]

print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
print(training_df.head(200))

print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
print(training_df.describe(include='all'))


#view correlation matrix, 1 means high correlation, -1 inverse correlation, 0 no correlation
print(training_df.corr(numeric_only = True))

#View pairplot, it shows graphs correlating variables
sns.pairplot(training_df, x_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"], y_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
plt.show()


# Pedi para que o deepseek fizesse esses comentários
# Função principal para criar os gráficos
# def make_plots(df, feature_names, label_name, model_output, sample_size=200):
#     # Cria uma amostra aleatória do DataFrame
#     random_sample = df.sample(n=sample_size).copy()
#     # Reseta o índice da amostra (não é salvo, pois não há atribuição)
#     random_sample.reset_index()
#     # Desempacota a saída do modelo (pesos, bias, épocas e RMSE)
#     weights, bias, epochs, rmse = model_output

#     # Verifica se o gráfico será 2D (uma feature) ou 3D (duas features)
#     is_2d_plot = len(feature_names) == 1
#     # Define o tipo de gráfico do modelo: scatter (2D) ou surface (3D)
#     model_plot_type = "scatter" if is_2d_plot else "surface"

#     # Cria uma figura com dois subplots: curva de perda e gráfico do modelo
#     fig = make_subplots(rows=1, cols=2,
#                         subplot_titles=("Loss Curve", "Model Plot"),
#                         specs=[[{"type": "scatter"}, {"type": model_plot_type}]])

#     # Adiciona os dados ao gráfico
#     plot_data(random_sample, feature_names, label_name, fig)
#     # Adiciona o modelo ao gráfico
#     plot_model(random_sample, feature_names, weights, bias, fig)
#     # Adiciona a curva de perda ao gráfico
#     plot_loss_curve(epochs, rmse, fig)

#     # Exibe a figura com os subplots
#     fig.show()
#     return

# # Função para plotar a curva de perda (RMSE ao longo das épocas)
# def plot_loss_curve(epochs, rmse, fig):
#     # Cria um gráfico de linha com as épocas no eixo X e RMSE no eixo Y
#     curve = px.line(x=epochs, y=rmse)
#     # Personaliza a linha (cor vermelha e espessura 3)
#     curve.update_traces(line_color='#ff0000', line_width=3)
#     # Adiciona o gráfico de linha ao subplot da curva de perda
#     fig.append_trace(curve.data[0], row=1, col=1)
#     # Define o título do eixo X como "Epoch"
#     fig.update_xaxes(title_text="Epoch", row=1, col=1)
#     # Define o título do eixo Y como "Root Mean Squared Error" e ajusta o intervalo
#     fig.update_yaxes(title_text="Root Mean Squared Error", row=1, col=1, range=[rmse.min()*0.8, rmse.max()])
#     return

# # Função para plotar os dados
# def plot_data(df, features, label, fig):
#     # Verifica se o gráfico será 2D ou 3D
#     if len(features) == 1:
#         # Cria um gráfico de dispersão 2D
#         scatter = px.scatter(df, x=features[0], y=label)
#     else:
#         # Cria um gráfico de dispersão 3D
#         scatter = px.scatter_3d(df, x=features[0], y=features[1], z=label)
#     # Adiciona o gráfico de dispersão ao subplot do modelo
#     fig.append_trace(scatter.data[0], row=1, col=2)
#     # Para gráficos 2D, define os títulos dos eixos X e Y
#     if len(features) == 1:
#         fig.update_xaxes(title_text=features[0], row=1, col=2)
#         fig.update_yaxes(title_text=label, row=1, col=2)
#     else:
#         # Para gráficos 3D, define os títulos dos eixos X, Y e Z
#         fig.update_layout(scene1=dict(xaxis_title=features[0], yaxis_title=features[1], zaxis_title=label))
#     return

# # Função para plotar o modelo
# def plot_model(df, features, weights, bias, fig):
#     # Inicializa a coluna 'FARE_PREDICTED' com o valor do bias
#     df['FARE_PREDICTED'] = bias[0]
#     # Calcula as previsões do modelo usando os pesos e as features
#     for index, feature in enumerate(features):
#         df['FARE_PREDICTED'] = df['FARE_PREDICTED'] + weights[index][0] * df[feature]
#     # Verifica se o gráfico será 2D ou 3D
#     if len(features) == 1:
#         # Cria uma linha representando o modelo para gráficos 2D
#         model = px.line(df, x=features[0], y='FARE_PREDICTED')
#         # Personaliza a linha (cor vermelha e espessura 3)
#         model.update_traces(line_color='#ff0000', line_width=3)
#     else:
#         # Para gráficos 3D, calcula os pontos para a superfície do modelo
#         z_name, y_name = "FARE_PREDICTED", features[1]
#         z = [df[z_name].min(), (df[z_name].max() - df[z_name].min()) / 2, df[z_name].max()]
#         y = [df[y_name].min(), (df[y_name].max() - df[y_name].min()) / 2, df[y_name].max()]
#         x = []
#         for i in range(len(y)):
#             x.append((z[i] - weights[1][0] * y[i] - bias[0]) / weights[0][0])
#         # Cria um DataFrame para a superfície do modelo
#         plane = pd.DataFrame({'x': x, 'y': y, 'z': [z] * 3})
#         # Define uma escala de cores para a superfície
#         light_yellow = [[0, '#89CFF0'], [1, '#FFDB58']]
#         # Cria uma superfície 3D usando plotly.graph_objects
#         model = go.Figure(data=go.Surface(x=plane['x'], y=plane['y'], z=plane['z'],
#                           colorscale=light_yellow))
#     # Adiciona o modelo ao subplot
#     fig.add_trace(model.data[0], row=1, col=2)
#     return

# # Função para exibir informações sobre o modelo
# def model_info(feature_names, label_name, model_output):
#     # Extrai pesos e bias da saída do modelo
#     weights = model_output[0]
#     bias = model_output[1]
#     # Cria um banner para exibir as informações do modelo
#     nl = "\n"
#     header = "-" * 80
#     banner = header + nl + "|" + "MODEL INFO".center(78) + "|" + nl + header
#     # Inicializa variáveis para armazenar informações e a equação do modelo
#     info = ""
#     equation = label_name + " = "
#     # Constrói a string de informações e a equação do modelo
#     for index, feature in enumerate(feature_names):
#         info = info + "Weight for feature[{}]: {:.3f}\n".format(feature, weights[index][0])
#         equation = equation + "{:.3f} * {} + ".format(weights[index][0], feature)
#     # Adiciona o bias às informações e à equação
#     info = info + "Bias: {:.3f}\n".format(bias[0])
#     equation = equation + "{:.3f}\n".format(bias[0])
#     # Retorna as informações formatadas
#     return banner + nl + info + nl + equation

# # Mensagem de sucesso
# print("SUCCESS: defining plotting functions complete.")

# #@title Code - Define ML functions

# def build_model(my_learning_rate, num_features):
#   """Create and compile a simple linear regression model."""
#   # Describe the topography of the model.
#   # The topography of a simple linear regression model
#   # is a single node in a single layer.
#   inputs = keras.Input(shape=(num_features,))
#   outputs = keras.layers.Dense(units=1)(inputs)
#   model = keras.Model(inputs=inputs, outputs=outputs)

#   # Compile the model topography into code that Keras can efficiently
#   # execute. Configure training to minimize the model's mean squared error.
#   model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=my_learning_rate),
#                 loss="mean_squared_error",
#                 metrics=[keras.metrics.RootMeanSquaredError()])

#   return model


# def train_model(model, df, features, label, epochs, batch_size):
#   """Train the model by feeding it data."""

#   # Feed the model the feature and the label.
#   # The model will train for the specified number of epochs.
#   # input_x = df.iloc[:,1:3].values
#   # df[feature]
#   history = model.fit(x=features,
#                       y=label,
#                       batch_size=batch_size,
#                       epochs=epochs)

#   # Gather the trained model's weight and bias.
#   trained_weight = model.get_weights()[0]
#   trained_bias = model.get_weights()[1]

#   # The list of epochs is stored separately from the rest of history.
#   epochs = history.epoch

#   # Isolate the error for each epoch.
#   hist = pd.DataFrame(history.history)

#   # To track the progression of training, we're going to take a snapshot
#   # of the model's root mean squared error at each epoch.
#   rmse = hist["root_mean_squared_error"]

#   return trained_weight, trained_bias, epochs, rmse


# def run_experiment(df, feature_names, label_name, learning_rate, epochs, batch_size):

#   print('INFO: starting training experiment with features={} and label={}\n'.format(feature_names, label_name))

#   num_features = len(feature_names)

#   features = df.loc[:, feature_names].values
#   label = df[label_name].values

#   model = build_model(learning_rate, num_features)
#   model_output = train_model(model, df, features, label, epochs, batch_size)

#   print('\nSUCCESS: training experiment complete\n')
#   print('{}'.format(model_info(feature_names, label_name, model_output)))
#   make_plots(df, feature_names, label_name, model_output)

#   return model

# print("SUCCESS: defining linear regression functions complete.")