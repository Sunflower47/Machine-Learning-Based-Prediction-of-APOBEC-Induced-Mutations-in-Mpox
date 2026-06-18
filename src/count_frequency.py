from collections import Counter
import pandas as pd
from ete3 import Tree

df = pd.read_csv("data/raw/branch_mutations.txt", sep="\t")

# select only SNPs
df_snp = df[(df["state1"] != "-") & (df["state2"] != "-")]

# select C>T (G>A) mutations
df_snp_filt = df_snp[
    ((df_snp.state1 == 'C')&(df_snp.state2.isin(['T','W','Y','K','B'])))|
     ((df_snp.state1 == 'G')&(df_snp.state2.isin(['A', 'W','M'])))]

df_mutations = pd.read_csv("data/interim/apobec_positions_list.csv", sep="\t")

df_pos = df_mutations[['pos', 'motif']]
pos_tc = df_pos[df_pos.motif == 'tc/tt'].pos + 2
pos_ga = df_pos[df_pos.motif == 'ga/aa'].pos + 1

df_positions_1 = pd.DataFrame({'pos':pos_tc, 'motif':len(pos_tc)*['tc/tt']})
df_positions_2 = pd.DataFrame({'pos':pos_ga, 'motif':len(pos_ga)*['ga/aa']})

df_positions = pd.concat([df_positions_1, df_positions_2])

df_snp_filt = df_snp_filt[df_snp_filt.pos.isin(df_positions.pos)]

df_snp_filt['counter'] = df_snp_filt.pos.apply(lambda x: Counter(df_snp_filt.pos)[x])

tree = Tree("data/raw/final_alignment_updated.afa.treefile", format=1)
node = tree.search_nodes(name='NC_063383.1')[0]
leaves = node.get_leaf_names()

df_snp_filt['descendants'] = df_snp_filt['node'].apply(lambda x: tree.search_nodes(name=x)[0].get_leaf_names())

positions_counter = df_snp_filt[['pos', 'state1', 'state2', 'counter']].drop_duplicates()
positions_counter.to_pickle('data/interim/positions_counter.pkl.gz', compression='gzip')
