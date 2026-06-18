from Bio import AlignIO
import numpy as np
from collections import Counter
import pandas as pd
import gzip

def process_motif_dataframe(matrix, positions, motif_name, opposite_patterns):
  """Select and filter sequence positions of dinucleotide motifs."""
  col1 = matrix[:, positions]
  col2 = matrix[:, positions + 1]
  stacked = np.char.add(col1, col2)

  df = pd.DataFrame({'pos': positions})
  df['motifs'] = list(stacked.T)
  df['counter'] = [Counter(x) for x in df['motifs']]

  p1, p2 = opposite_patterns
  df_filtered = df[
      df['counter'].apply(
          lambda x: (x[p1] + x[p2]) >= matrix.shape[0] - 21
      )].copy()

  df_filtered_ = df_filtered[
      df_filtered['counter'].apply(
          lambda x: False if (x[p1] == 0) | (x[p2] == 0) else True
      )].copy()

  df_filtered_['motif'] = motif_name
  return df, df_filtered_

path = 'data/raw/final_alignment_updated.afa.gz'

with gzip.open(path, "rt") as handle:
    alignment = AlignIO.read(handle, "fasta")
matrix = np.array([list(rec.seq) for rec in alignment])

dinucleotides = np.char.add(matrix[:, :-1], matrix[:, 1:])
mask_tc_tt = (dinucleotides == 'tc') | (dinucleotides == 'tt')
mask_ga_aa = (dinucleotides == 'ga') | (dinucleotides == 'aa')

all_pos_tc_tt = np.where(np.any(mask_tc_tt, axis=0))[0]
all_pos_ga_aa = np.where(np.any(mask_ga_aa, axis=0))[0]

df_tc_tt, df_tc_tt_filt = process_motif_dataframe(
        matrix, all_pos_tc_tt, 'tc/tt', ('tc', 'tt'))
df_ga_aa, df_ga_aa_filt = process_motif_dataframe(
        matrix, all_pos_ga_aa, 'ga/aa', ('ga', 'aa'))

df_final = pd.concat([df_tc_tt_filt, df_ga_aa_filt], ignore_index=True)

df_final['pos_end'] = df_final['pos'] + 1
df_final['counter'] = df_final['counter'].apply(
            lambda x: {str(k): v for k, v in x.items()})
df_final[['pos', 'pos_end', 'motif', 'counter']].to_csv(
            'data/interim/apobec_positions_list.csv', index=False, sep='\t')

df_tc_tt = df_tc_tt.copy()
df_ga_aa = df_ga_aa.copy()

df_tc_tt['motif'] = 'tc/tt'
df_ga_aa['motif'] = 'ga/aa'

df_tc_tt['counter'] = df_tc_tt['counter'].apply(
    lambda x: {str(k): v for k, v in x.items()})
df_ga_aa['counter'] = df_ga_aa['counter'].apply(
    lambda x: {str(k): v for k, v in x.items()})

pd.concat([df_tc_tt, df_ga_aa], ignore_index=True).to_csv(
        'data/interim/all_positions_list.csv', index=False, sep='\t')
