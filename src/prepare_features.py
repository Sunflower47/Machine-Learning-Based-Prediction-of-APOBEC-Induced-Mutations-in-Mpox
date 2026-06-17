import numpy as np
import pandas as pd

df = pd.read_pickle('data/interim/input_encoded.pkl.gz')
df = df.reset_index(drop=True)

df_pos_counter = pd.read_pickle('data/interim/positions_counter.plk.gz')

df_reg = df[df.label == 0].copy()
df_reg['counter'] = df_reg.label.copy()

df_mutated = df_pos_counter.sort_values(by='pos')[['pos', 'counter', 'state1']]

df_mutated.loc[df_mutated['state1'] == 'C', 'pos'] -= 2
df_mutated.loc[df_mutated['state1'] == 'G', 'pos'] -= 1

df_mutated = df_mutated.drop_duplicates(subset=['pos'])

df_input = pd.concat([df_reg, pd.merge(df_mutated[['pos', 'counter']], df[df.label == 1], on='pos', how='inner')])

X = np.stack(df_input.seq_encoded)
y = df_input.label.to_numpy()

Z = df_input.counter.to_numpy()

X_masked = X.copy()

X_masked[:, 80:82, :] = 0

X_50 = X_masked[:, 30:132, :].reshape((X_masked.shape[0], 102*4))
X_30 = X_masked[:, 30+20:132-20, :].reshape((X_masked.shape[0], 62*4))

data_to_save = {
    'X_50': X_50,
    'X_30': X_30,
    'y': y,
    'Z': Z,
    'pos': df_input.pos.to_numpy()
}

np.savez_compressed('data/processed/models_input.npz', **data_to_save)
