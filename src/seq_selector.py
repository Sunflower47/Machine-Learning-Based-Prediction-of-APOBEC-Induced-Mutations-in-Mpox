from Bio import AlignIO, SeqIO
from Bio.Seq import Seq

import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from collections import Counter
import ast


def get_sequence_around_pos(matrix, pos, n):
  """Extract sequence region around a specific position."""
  return matrix[:, pos-n:pos+n+2]


def reverse_complement_matrix(matrix):
  """Computes the reverse complement for a matrix of biological sequences."""
  reverse_complement_matrix = []
  for i in range(matrix.shape[0]):
      reverse_complement_matrix.append(list(Seq(''.join(matrix[i])).reverse_complement()))
  return np.array(reverse_complement_matrix)



def remove_row(arr, idx):
  """Remove a specific row from the matrix."""
  return np.vstack([arr[:idx], arr[idx+1:]])

path = 'final_alignment_updated.afa'
alignment = AlignIO.read(path, "fasta")
matrix = np.array([list(rec.seq) for rec in alignment])

target_id = 'NC_063383.1' #reference genome
row_index = [rec.id for rec in alignment].index(target_id)

df_positions = pd.read_csv('all_positions_list.csv', sep='\t')

#Filter sequences around TC motifs
df_positions.counter = df_positions.counter.apply(lambda x: Counter(ast.literal_eval(x)))
df_tc = df_positions[df_positions.motif == 'tc/tt']

#Allow other mutations to occur at most 20 times
df_tc_filtered = df_tc[df_tc.counter.apply(lambda x: True if (x['tt'] == 0) else False)]
df_tc_filtered = df_tc_filtered[df_tc_filtered.counter.apply(
    lambda x: True if x['tc'] >= matrix.shape[0]-20 else False)]

#Drop positions with double gaps in the motif area
df_tc_filtered = df_tc_filtered[df_tc_filtered.counter.apply(lambda x: True if (x['--'] == 0) else False)]


#Drop positions with single gaps ('t-' or '-c')
df_tc_filtered = df_tc_filtered[df_tc_filtered.counter.apply(
    lambda x: True if (x['t-'] == 0)|(x['-c'] == 0) else False)]

#Filter positions by genome boundaries to avoid edges
df_tc_filtered = df_tc_filtered[(df_tc_filtered.pos > 600)&(df_tc_filtered.pos < 191500)]

#Extract context window of size n around the motif
n = 80
df_tc_filtered['seq'] = df_tc_filtered.pos.apply(lambda x: get_sequence_around_pos(matrix, x, n))

#Filter sequences around GA motifs
#Apply the exact same filtering steps as for the 'tc' motif

df_ga = df_positions[df_positions.motif == 'ga/aa']

df_ga_filtered = df_ga[df_ga.counter.apply(lambda x: True if (x['aa'] == 0) else False)]
df_ga_filtered = df_ga_filtered[df_ga_filtered.counter.apply(
    lambda x: True if x['ga'] >= matrix.shape[0]-20 else False)]

df_ga_filtered = df_ga_filtered[df_ga_filtered.counter.apply(lambda x: True if (x['--'] == 0) else False)]

df_ga_filtered = df_ga_filtered[df_ga_filtered.counter.apply(
    lambda x: True if (x['-a'] == 0)|(x['g-'] == 0) else False)]

df_ga_filtered = df_ga_filtered[(df_ga_filtered.pos > 600)&(df_ga_filtered.pos < 191500)]

df_ga_filtered['seq'] = df_ga_filtered.pos.apply(lambda x: get_sequence_around_pos(matrix,x,n))


#Convert sequences to reverse complements
df_ga_filtered['seq_reverse_complement'] = df_ga_filtered.seq.apply(reverse_complement_matrix)
df_ga_filtered_ = df_ga_filtered[['pos', 'motifs', 'counter', 'motif', 'seq_reverse_complement']]
df_ga_filtered_ = df_ga_filtered_.rename(columns={'seq_reverse_complement': 'seq'})
tc = pd.concat([df_ga_filtered_, df_tc_filtered])


# Remove the reference genome (unaffected by APOBEC mutations) from the extracted non-mutated sequences
seq_filtered = np.zeros((tc.shape[0], remove_row(tc.seq.iloc[8], row_index).shape[0],remove_row(tc.seq.iloc[8], row_index).shape[1]), dtype='U1')

for i in range(tc.shape[0]):
    seq_filtered[i] = remove_row(tc.seq.iloc[i], row_index)

tc['seq_filtered'] = list(seq_filtered)

tc[['pos', 'motifs', 'counter', 'motif', 'seq_filtered']].to_pickle('non_mutated_positions_'+str(n)+'.pkl.gz', compression='gzip')

#Filter mutated sequences
df_tc_tt = pd.read_csv('apobec_positions_list.csv', sep='\t')
df_tc_tt = df_tc_tt[(df_tc_tt.pos > 600)&(df_tc_tt.pos < 191500)]
df_tc_tt['seq'] = df_tc_tt.pos.apply(lambda x: get_sequence_around_pos(matrix,x,n))
df_tc_tt['seq'] = df_tc_tt.apply(lambda x: x.seq if x.motif=='tc/tt' else reverse_complement_matrix(x.seq), axis=1)
seq_filtered_tctt = np.zeros((df_tc_tt.shape[0],
                              remove_row(df_tc_tt.seq.iloc[1], row_index).shape[0],
                              remove_row(df_tc_tt.seq.iloc[1], row_index).shape[1]), dtype='U1')

for i in range(df_tc_tt.shape[0]):
    seq_filtered_tctt[i] = remove_row(df_tc_tt.seq.iloc[i], row_index)

df_tc_tt['seq_filtered'] = list(seq_filtered_tctt)

df_tc_tt[['pos', 'counter', 'motif', 'seq_filtered']].to_pickle('mutated_positions_'+str(n)+'.pkl.gz', compression='gzip')
