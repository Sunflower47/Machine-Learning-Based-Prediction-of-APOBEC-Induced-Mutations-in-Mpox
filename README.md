# 🦠 Project Goal
To develop machine learning models for predicting APOBEC-induced mutations at TC motif sites in monkeypox virus genomes based on their flanking nucleotide context.


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
## 1. Logistic Regression
| Window | Training Regime | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **50 bp** | Baseline | 0.69 | 0.19 | **0.16** | 0.61 | 0.25 |
| | Weighted | 0.69 | 0.19 | 0.12 | **0.79** | 0.22 |
| | With Additional Features | **0.70** | **0.22** | **0.16** | 0.63 | **0.26** |
| **30 bp** | Baseline | 0.70 | 0.21 | **0.16** | 0.63 | 0.25 |
| | Weighted | 0.70 | 0.21 | 0.12 | **0.82** | 0.22 |
| | With Additional Features | **0.71** | **0.23** | **0.16** | 0.64 | **0.26** |

> The final ROC and PR curves can be found in the corresponding `.ipynb` files.

For both flanking window sizes, the nucleotides closest to the motif (1 base pair upstream and 3 base pairs downstream) had the highest impact on the model's predictions.

<table>
  <tr>
    <td colspan="2" align="center"><strong>Logistic Regression Weights-Based Sequence Logo (50 bp Flanking Window)</strong></td>
  </tr>
  <tr>
    <td align="center"><strong>Mutation-Prone Context</strong></td>
    <td align="center"><strong>Stable Control Context</strong></td>
  </tr>
  <tr>
    <td style="text-align: center;">
          <img width="590" height="340" alt="image" src="https://github.com/user-attachments/assets/b42572f1-885c-4abf-9c25-761849928b3c" />
    </td>
    <td style="text-align: center;">
          <img width="590" height="340" alt="image" src="https://github.com/user-attachments/assets/d55f50a7-130b-4dbc-91ce-c6aa00f590d8" />
    </td>
</table>

## 2. XGBoost
| Window | Training Regime | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **50 bp** | Baseline | 0.69 | 0.19 | 0.16 | 0.60 | 0.25 |
| | Weighted | 0.68 | 0.20 | 0.15 | **0.64** | 0.24 |
| | With Additional Features | **0.72** | **0.23** | **0.18** | 0.61 | **0.28** |
| **30 bp** | Baseline | 0.69 | 0.19 | 0.16 | 0.61 | 0.25 |
| | Weighted | 0.67 | 0.18 | 0.14 | **0.64** | 0.23 |
| | With Additional Features | **0.72** | **0.21** | **0.19** | 0.58 | **0.28** |

> The final ROC and PR curves can be found in the corresponding `.ipynb` files.

The feature importance analysis for XGBoost demonstrated that the immediate flanking nucleotides closest to the target motif carry the highest predictive weight.

<table>
  <tr>
    <td style="text-align: center;">
      <img width="850" height="390" alt="XGBoost Feature Importance Heatmap 30bp" src="https://github.com/user-attachments/assets/81162989-7037-494d-a654-e5f60bfae703" />
    </td>
  </tr>
  <tr>
    <td align="center"><strong>XGBoost Feature Importance Heatmap (30 bp Flanking Window)</strong></td>
  </tr>
</table>


Based on the **SHAP analysis**, the model successfully captured complex degenerate motif patterns within the flanking regions that influence APOBEC activity:
* **Decreased Mutation Susceptibility:** A lower probability of mutation is characteristically associated with sequence contexts matching the **`RTCYYW`** motif framework. 
* **Increased Mutation Risk:** Conversely, the presence of the **`YTCRRS`** pattern is interpreted by the model as a strong positive driver, significantly elevating the predicted likelihood of a mutational event.

> *The SHAP summary plots are fully documented and can be reviewed inside the XGBoost_model.ipynb.*

# 🧠 Deep Learning Approach
🛠️ **Overfitting Mitigation & Checkpointing:** To prevent overfitting, an early stopping mechanism with model checkpointing was implemented, tracking the best performance metric on the validation set and saving the optimal weights. However, this strategy was bypassed for the **Weighted Mode** regime, as the heavy class penalty coefficients introduced significant gradient instability during training, making standard validation-loss tracking unreliable.
🧬 **Strand-Aware Data Augmentation:** To leverage the double-stranded nature of viral DNA and expand the model's robustness, data augmentation was performed by incorporating predictions from both the forward and reverse-complementary strands during the evaluation phase.

## 3. CNN

| Window | Training Regime | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **50 bp** | Baseline | 0.63 | 0.16 | 0.14 | 0.54 | 0.22 |
| | Frequency-Aware (Weighted) | 0.61 | 0.14 | 0.11 | **0.74** | 0.19 |
| | Feature-Enriched | **0.67** | **0.20** | **0.14** | 0.65 | **0.23** |
| **30 bp** | Baseline | 0.64 | 0.16 | 0.13 | 0.62 | 0.21 |
| | Frequency-Aware (Weighted) | 0.63 | 0.15 | 0.11 | **0.88** | 0.19 |
| | Feature-Enriched | **0.67** | **0.20** | **0.14** | 0.67 | **0.24** |
## 4. Attention-based CNN

| Window | Training Regime | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **50 bp** | Baseline | 0.63 | 0.15 | 0.13 | 0.55 | 0.21 |
| | Frequency-Aware (Weighted) | 0.62 | 0.15 | 0.10 | **0.89** | 0.18 |
| | Feature-Enriched | **0.67** | **0.19** | **0.14** | 0.64 | **0.22** |
| **30 bp** | Baseline | 0.63 | 0.17 | 0.13 | 0.63 | 0.21 |
| | Frequency-Aware (Weighted) | 0.63 | 0.16 | 0.11 | **0.79** | 0.19 |
| | Feature-Enriched | **0.67** | **0.20** | **0.14** | 0.73 | **0.23** |

