import numpy as np
import networkx as nx
import netrd
import matplotlib.pyplot as plt
import itertools as it
import pandas as pd
import netrd.reconstruction
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class GreyRelationalAnalysis:
    def __init__(self):
        self.results = {}

    def fit(self, data):
        data = np.array(data)
        data_transposed = data.T
        scaler = StandardScaler()
        data_scaled = data_transposed
        num_sequences = data_scaled.shape[1]
        grey_matrix = np.zeros((num_sequences, num_sequences))
        max_diff_global = np.max(data_scaled) - np.min(data_scaled)
        rho = 0.8
        for i in range(num_sequences):
            for j in range(num_sequences):
                if i != j:
                    delta = np.abs(data_scaled[:, i] - data_scaled[:, j])
                    grey_matrix[i, j] = np.mean((rho * max_diff_global) / (delta + rho * max_diff_global))
                else:
                    grey_matrix[i, j] = 1
        self.results['thresholded_matrix'] = grey_matrix

dataset = "ASUh.csv"
df = pd.read_csv(dataset, header=0)

season = {
    'Spring'
    # 'Summer'
    # 'Autumn'
}
year = 2018
month = [3,4,5]
# month = [6,7,8]
# month = [9,10,11]


df = df[(df['Year'] == year) & (df['Month'].isin(month))]

df = df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Day_Type'])
df.fillna(df.mean(), inplace=True)
df_scaled = df.copy()

all_matrices_correlation = []
all_matrices_redundancy = []

recons_correlation = {
    'CorrelationMatrix': netrd.reconstruction.CorrelationMatrix(),
    'MarchenkoPastur': netrd.reconstruction.MarchenkoPastur(),
    'GrangerCausality': netrd.reconstruction.GrangerCausality(),
    'OUInference': netrd.reconstruction.OUInference(),
    'ThoulessAndersonPalmer': netrd.reconstruction.ThoulessAndersonPalmer(),
    # 'GreyRelationalAnalysis': GreyRelationalAnalysis(),
}

recons_redundancy = {
    'NaiveTransferEntropy': netrd.reconstruction.NaiveTransferEntropy(),
    'MutualInformationMatrix': netrd.reconstruction.MutualInformationMatrix(),
}

for ri, R1 in recons_correlation.items():
    try:
        R1.fit(np.array(df_scaled.T))
        adj = pd.DataFrame(R1.results['thresholded_matrix'])
        adj.replace([np.inf, np.nan], 0.0, inplace=True)
        adj[np.abs(adj) < 0.1] = 0
        all_matrices_correlation.append(adj.values)
    except Exception as e:
        print(f"Error with {ri}: {e}")
        continue

stacked_matrices = np.stack(all_matrices_correlation)
abs_stacked_matrices = np.abs(stacked_matrices)
indices_of_max = np.argmax(abs_stacked_matrices, axis=0)
fused_correlation_matrix = np.take_along_axis(stacked_matrices, indices_of_max[None, :, :], axis=0)[0]
pd.DataFrame(fused_correlation_matrix).to_csv(dataset.split(".")[0] + "_winter_fused1.csv", index=False)

for ri, R1 in recons_redundancy.items():
    try:
        R1.fit(np.array(df_scaled.T))
        adj = pd.DataFrame(R1.results['thresholded_matrix']).abs()
        adj.replace([np.inf, np.nan], 0.0, inplace=True)
        all_matrices_redundancy.append(adj.values)
    except Exception as e:
        print(f"Error with {ri}: {e}")
        continue

    stacked_matrices_redundancy = np.stack(all_matrices_redundancy)
    abs_stacked_matrices_redundancy = np.abs(stacked_matrices_redundancy)
    indices_of_redundancy_max = np.argmax(abs_stacked_matrices_redundancy, axis=0)
    fused_redundancy_matrix = np.take_along_axis(stacked_matrices_redundancy, indices_of_redundancy_max[None, :, :], axis=0)[0]

    pd.DataFrame(fused_redundancy_matrix).to_csv(dataset.split(".")[0] + f"_{season}_fused2.csv", index=False)

    final_matrix = fused_correlation_matrix.copy()
    mask = (np.abs(fused_correlation_matrix) >= 0.1) & (np.abs(fused_correlation_matrix) <= 0.5)
    redundancy_threshold = np.mean(fused_redundancy_matrix)  # 0.015
    print(f"Redundancy Threshold: {redundancy_threshold}")
    final_matrix[mask & (np.abs(fused_redundancy_matrix) < redundancy_threshold)] = 0

    pd.DataFrame(final_matrix).to_csv(dataset.split(".")[0] + f"_{season}_final_matrix.csv", index=False)

interest_columns_indices = [0, 1, 2, 3] 
n_rows, n_cols = final_matrix.shape

for col_index in range(n_cols):
    if col_index not in interest_columns_indices:
        if not np.any(final_matrix[interest_columns_indices, col_index]):
            final_matrix[:, col_index] = 0
            final_matrix[col_index, :] = 0

pd.DataFrame(final_matrix).to_csv(dataset.split(".")[0] + f"_{season}_final_matrix_cleaned.csv", index=False)

np.fill_diagonal(final_matrix, 1)

for i in range(n_rows):
    for j in range(i+1, n_cols):  
        if abs(final_matrix[i, j]) > abs(final_matrix[j, i]):
            max_val = final_matrix[i, j]
        else:
            max_val = final_matrix[j, i]
        final_matrix[i, j] = max_val
        final_matrix[j, i] = max_val

final_matrix = np.tril(final_matrix)

pd.DataFrame(final_matrix).to_csv(dataset.split(".")[0] + f"_{season}_final_matrix_triangular.csv", index=False)

final_matrix_five_columns = final_matrix[:, 0:4]

pd.DataFrame(final_matrix_five_columns).to_csv(dataset.split(".")[0] + f"_{season}_final_matrix.csv", index=False)

