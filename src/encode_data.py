import numpy as np
import pandas as pd

one_hot_encoder = {
    'A': np.array([1, 0, 0, 0]),
    'C': np.array([0, 1, 0, 0]),
    'G': np.array([0, 0, 1, 0]),
    'T': np.array([0, 0, 0, 1]),
    'U': np.array([0, 0, 0, 1]),
    'R': np.array([0.5, 0, 0.5, 0]),   # A or G
    'Y': np.array([0, 0.5, 0, 0.5]),   # C or T
    'S': np.array([0, 0.5, 0.5, 0]),   # G or C
    'W': np.array([0.5, 0, 0, 0.5]),   # A or T
    'K': np.array([0, 0, 0.5, 0.5]),   # G or T
    'M': np.array([0.5, 0.5, 0, 0]),   # A or C
    'B': np.array([0, 1, 1, 1])/3, # C, G, T
    'D': np.array([1, 0, 1, 1])/3, # A, G, T
    'H': np.array([1, 1, 0, 1])/3, # A, C, T
    'V': np.array([1, 1, 1, 0])/3, # A, C, G
    'N': np.array([0.25, 0.25, 0.25, 0.25]),
    '-':np.array([0, 0, 0, 0])}

mapping_matrix = np.zeros((256, 4))
for char, vector in one_hot_encoder.items():
    mapping_matrix[ord(char.upper())] = vector
    mapping_matrix[ord(char.lower())] = vector

def encode_sequence(seq):
  """Encoding of sequences to PWM"""
  seq_ascii = np.asarray(seq, dtype='S1').view(np.uint8)
  encoded_seqs = mapping_matrix[seq_ascii]
  pwm = encoded_seqs.mean(axis=0)
  consensus_indices = np.argmax(pwm, axis=1)
  consensus_encoded = np.eye(4)[consensus_indices]
  return consensus_encoded


df_not_mutated = pd.read_pickle('non_mutated_positions_80.pkl.gz')
df_mutated = pd.read_pickle('mutated_positions_80.pkl.gz')

df_not_mutated['seq_encoded'] = df_not_mutated['seq_filtered'].apply(encode_sequence)
df_mutated['seq_encoded'] = df_mutated['seq_filtered'].apply(encode_sequence)

df_not_mutated['label'] = np.zeros(len(df_not_mutated), dtype=int)
df_mutated['label'] = np.ones(len(df_mutated), dtype=int)

df = pd.concat([df_mutated[['pos', 'seq_encoded', 'label']], df_not_mutated[['pos', 'seq_encoded', 'label']]])
df = df.sort_values(by='pos')


df.to_pickle('input_encoded.pkl.gz', compression='gzip')
