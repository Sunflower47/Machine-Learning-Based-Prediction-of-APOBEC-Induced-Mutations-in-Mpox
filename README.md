# 🦠 Project Goal
To develop machine learning models for predicting APOBEC-induced mutations at TC motif sites within monkeypox virus genomes based on their flanking nucleotide context.


# 🧬 Data Pipeline & Preprocessing

1. Selecting viral genomic sequences (MPOX) from the GISAID database (3,201 genomes).
2. Similarity-based clustering using `Mash` and `FastANI`.
3. Aligning genomic sequence blocks using `MAFFT`.
4. Combining the aligned blocks into a single global alignment using the `mafft --merge` mode.
5. Estimating true mutation frequencies independent of lineage inheritance using `IQ-TREE` and `TreeTime` for ancestral state reconstruction.
6. Extracting `TC` (`GA`) motif positions and splitting them into 21,148 target mutation sites TC>TT (GA>AA) and 2,167 non-mutated control sites.
7. Deriving the consensus sequence from the global alignment.
8. Applying One-Hot Encoding to the flanking context and masking the target motif positions with zeros to prevent data leakage.
9. Integrating additional biological features into the model, including DNA secondary structure parameters at the moment of enzyme activity and Grantham scores for the resulting amino acid substitutions.

# 🤖 Machine Learning Framework

## Training Strategies, Window Sizes & Class Imbalance
To analyze the impact of genomic context length, all experiments were conducted using two distinct flanking window sizes: **30 bp** and **50 bp** on each side of the target motif. 
Additionally, to address the severe class imbalance in the dataset, **class weighting** was implemented across all architectures to ensure the minority class is heavily penalized during training. Every model was evaluated under three progressive training regimes across both window sizes:

1. **Baseline:** Trained strictly on One-Hot Encoded flanking sequences with masked target motifs.
2. **Weighted:** Baseline features integrated with true mutation frequencies.
3. **With Additional Features:** Nucleotide context combined with DNA secondary structure parameters and Grantham scores.

---


