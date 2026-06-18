import numpy as np
from collections import Counter
import pandas as pd
from Bio import AlignIO
from Bio.Seq import Seq
import gzip

path = 'data/raw/final_alignment_updated.afa.gz'

with gzip.open(path, "rt") as handle:
    alignment = AlignIO.read(handle, "fasta")
matrix = np.array([list(rec.seq) for rec in alignment])

target_id = 'NC_063383.1'
row_index = [rec.id for rec in alignment].index(target_id)

seq_ref = matrix[row_index]

dict_seq_ref_align = {i:nucl for i,nucl in enumerate(seq_ref)}

keys = np.array(list(dict_seq_ref_align.keys()))
val = np.array(list(dict_seq_ref_align.values()))

ref_to_align = {int(i):int(k) for k,i in zip(keys[val != '-'], np.arange(len(val[val != '-'])))}
align_to_ref = {int(k):int(i) for k,i in zip(keys[val != '-'], np.arange(len(val[val != '-'])))}

annotation = pd.read_csv("data/raw/GCF_014621545.1_ASM1462154v1_genomic.gtf", sep='\t', comment='#', skiprows=3,
                         names= ['name', 'database', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attributes']).sort_values(by='start')

genes = annotation[annotation.feature == 'gene'].copy()
start_codons = annotation[annotation.feature == 'start_codon'].copy()

genes['start'] -= 1
genes['end'] -= 1

start_codons['start'] -= 1
start_codons['end'] -= 1

df_input = pd.read_pickle('data/interim/input_encoded.pkl.gz')
df_pos = df_input[['pos', 'seq_encoded', 'label']]
df_pos['nucl'] = df_pos.pos.apply(lambda x: str(Counter(matrix[:, x]).most_common(1)[0][0]))

def get_ref_pos(row, alignment_map):
    pos = row['pos'] if row['nucl'] == 'g' else row['pos'] + 1
    return alignment_map.get(pos)

df_pos['pos_in_ref'] = df_pos.apply(get_ref_pos, axis=1, args=(align_to_ref,))

df_pos['isin_gene'] = df_pos.pos_in_ref.apply(lambda x: any((x >= genes.start)&(x <= genes.end)))

df_in_genes = df_pos[df_pos['isin_gene'] == True].copy()

df_in_genes['strand'] = df_in_genes.pos_in_ref.apply(lambda x: genes[(x >= genes.start)&(x <= genes.end)].strand.iloc[0])
df_in_genes['start'] = df_in_genes.pos_in_ref.apply(lambda x: genes[(x >= genes.start)&(x <= genes.end)].start.iloc[0])
df_in_genes['end'] = df_in_genes.pos_in_ref.apply(lambda x: genes[(x >= genes.start)&(x <= genes.end)].end.iloc[0])

df_in_genes = df_in_genes.reset_index(drop=True)

def return_codon(row, matrix):
    position = row.pos_in_ref
    strand = row.strand
    if strand == '+':
        start = row.start
        in_codon = abs(position - start) % 3
        s = position - in_codon
        to_extract_ = np.array([s, s+1, s+2])
        to_extract = np.array([ref_to_align[x] for x in to_extract_])
        codon_list = list(Counter([''.join(row) for row in matrix[:, to_extract]]).most_common(1)[0][0])
        mutated_codon_list = codon_list.copy()
        if row.nucl == 't':
          codon_list[in_codon] = 'c'
          mutated_codon_list[in_codon] = 't'
        else:
          codon_list[in_codon] = 'g'
          mutated_codon_list[in_codon] = 'a'
        return codon_list, mutated_codon_list

    else:
        start = row.end
        in_codon = abs(position - start) % 3
        s = position - (2 - in_codon)
        to_extract_ = np.array([s, s+1, s+2])
        to_extract = np.array([ref_to_align[x] for x in to_extract_])
        codon_str = Counter([''.join(row) for row in matrix[:, to_extract]]).most_common(1)[0][0][::-1]
        codon_list = list(Seq(codon_str).complement())
        mutated_codon_list = codon_list.copy()
        if row.nucl == 't':
          codon_list[in_codon] = 'g'
          mutated_codon_list[in_codon] = 'a'
        else:
          codon_list[in_codon] = 'c'
          mutated_codon_list[in_codon] = 't'
        return codon_list, mutated_codon_list

df_in_genes['codon'] = df_in_genes.apply(lambda x: return_codon(x, matrix), axis=1)
df_in_genes['aa'] = df_in_genes.codon.apply(lambda x: [str(Seq(''.join(x[0])).translate()), str(Seq(''.join(x[1])).translate())])

def is_start(row):
    position = row.pos_in_ref
    strand = row.strand
    if strand == '+':
      return position <= row.start + 2
    else:
      return position >= row.end - 2

df_in_genes['is_start'] = df_in_genes.apply(lambda x: is_start(x), axis=1)

score_matrix = pd.read_csv('data/raw/grantham.tsv', sep='\t', index_col=0)

all_amino_acids = list(set(score_matrix.index) | set(score_matrix.columns))

full_matrix = pd.DataFrame(0, index=all_amino_acids, columns=all_amino_acids)

for row_aa in score_matrix.index:
    for col_aa in score_matrix.columns:
        val = score_matrix.loc[row_aa, col_aa]
        if val > 0:
            full_matrix.loc[row_aa, col_aa] = val
            full_matrix.loc[col_aa, row_aa] = val


def get_grantham_safe(pair, matrix):
    if pair[0] == '*' and pair[1] == '*':
        return 0

    if pair[0] not in matrix.index or pair[1] not in matrix.columns:
        return None


    return matrix.at[pair[0], pair[1]]


df_in_genes['score'] = df_in_genes.apply(
    lambda x: None if x.is_start else get_grantham_safe(x.aa, full_matrix),
    axis=1)

df_in_genes['stop_label'] = df_in_genes.score.isna().astype(int)

df_in_genes[['pos', 'label', 'codon', 'aa', 'score', 'stop_label']].to_pickle('data/processed/grantham_score.pkl.gz')

hairpin_results = pd.read_csv('data/raw/hairpin_results.csv')

hairpin_results['pos_in_align'] = hairpin_results.Position.apply(lambda x: ref_to_align[x-1])

hairpin_results_filt = hairpin_results[['Position', 'pos_in_align', 'Free_Energy_kcal_mol', 'Stem_Length', 'Loop_Length', 'Nucleotide_After_Motif']]

add_features = hairpin_results_filt[hairpin_results_filt.pos_in_align.isin(df_input['pos'])].copy()

add_features.to_pickle('data/processed/dataframe_features_nan.pkl.gz')

add_features['energy_isna'] = add_features.Free_Energy_kcal_mol.apply(lambda x: int(np.isnan(x)))
add_features['stem_isna'] = add_features.Stem_Length.apply(lambda x: int(np.isnan(x)))
add_features['loop_isna'] = add_features.Loop_Length.apply(lambda x: int(np.isnan(x)))

add_features['Free_Energy_kcal_mol'] = add_features.Free_Energy_kcal_mol.fillna(add_features.Free_Energy_kcal_mol.median())
add_features['Stem_Length'] = add_features.Stem_Length.fillna(add_features.Stem_Length.median())
add_features['Loop_Length'] = add_features.Loop_Length.fillna(add_features.Loop_Length.median())

add_features.to_pickle('data/processed/dataframe_features.pkl.gz')
