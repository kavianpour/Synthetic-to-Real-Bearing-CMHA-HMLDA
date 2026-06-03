# Method — Convolutional Multi-Head Attention + Hybrid Multi-Layer Domain Alignment (CMHA-HMLDA)

This document describes CMHA-HMLDA in full: the CNN + multi-head attention feature extractor, the MK-MMD and LMMD alignment losses, the objective function, and the training configuration reported in the paper.

---

## 1. Overview

CMHA-HMLDA is an **unsupervised synthetic-to-real** framework for bearing fault diagnosis. Labeled **synthetic** data is the source domain; **unlabeled real** data is the target. It has:

1. A **feature extractor** = five CNN layers + a multi-head attention block.
2. A **hybrid multi-layer domain alignment** loss = global (MK-MMD) + local (LMMD), applied across multiple FC layers.
3. A **classifier** (SoftMax head) trained on the labeled source.

Source data $D_S = \{(x_i^s, y_i^s)\}$ is labeled; target data is unlabeled. Both share a feature space ($X^s, X^t \in \chi$) but have different marginal distributions ($P^s(X^s) \neq P^t(X^t)$).

---

## 2. Feature extractor

### 2.1 CNN backbone

Five CNN layers form the backbone, each: Conv1D → batch normalization → **SELU** activation → max pooling (the fifth uses adaptive max pooling). SELU is used instead of ReLU because it reduces overfitting risk and carries a self-normalizing property. The extractor $F(\cdot)$ takes input $(X_i^s, X_j^t)$ and returns features $(H_i^s, H_j^t)$.

### 2.2 Multi-head attention

Scaled dot-product attention:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where $Q, K, V$ are the query, key, value matrices. To observe multiple sub-spaces by linearly projecting $Q, K, V$ at $h$ different positions, a multi-head mechanism is used:

$$\mathrm{MultiHead}(Q, K, V) = \mathrm{concat}(\mathrm{Head}_1, \dots, \mathrm{Head}_h)\,\theta^0$$

Multi-head attention combined with CNNs attends to more robust information than single-head attention at the same computational cost.

---

## 3. Domain alignment losses

### 3.1 Global — Maximum Mean Discrepancy (MMD / MK-MMD)

MMD measures the global discrepancy between distributions in a reproducing kernel Hilbert space (RKHS). For $n_s = n_t$:

$$D_k^2(X^s, X^t) = \left\lVert \tfrac{1}{n_1}\sum_{i=1}^{n_1} f(x_i^s) - \tfrac{1}{n_2}\sum_{j=1}^{n_2} f(x_j^t) \right\rVert_{\mathcal{H}}^2$$

To better match distributions, **MK-MMD** uses an optimal kernel that is a linear combination of multiple kernels:

$$K \triangleq \left\{ K = \sum_{\alpha=1}^{m} \beta_\alpha k_u : \beta_\alpha \geq 0,\ \forall \alpha \right\}$$

### 3.2 Local — Local Maximum Mean Discrepancy (LMMD)

LMMD divides the global distribution into $C$ sub-domains (one per class) and reduces the discrepancy between relevant sub-domains:

$$D(X^s, X^t)_{\mathcal{H}} = \frac{1}{C}\sum_{c=1}^{C} \left\lVert \sum_{x_i \in D_s} w_i^{sc}\,\phi(x_i^s) - \sum_{x_j \in D_t} w_j^{tc}\,\phi(x_j^t) \right\rVert_{\mathcal{H}}^2$$

with weights $\sum_i w_i^{sc} = \sum_j w_j^{tc} = 1$. Because LMMD weights require labels, **target pseudo-labels** (and source pseudo-labels for classification) are produced from the feature extractor via argmax over the SoftMax output.

---

## 4. Objective function

The classification loss on the labeled source:

$$\mathcal{L}_C(H_i^s, y_i^s) = \mathbb{E}_{(x^s, s)}\, L(\hat{y}_i^s, y_i^s)$$

The hybrid loss combines global and local terms, and the full objective is:

$$\mathcal{L}_{object}(H_i^s, H_i^t, y_i^s) = \lambda_1 \mathcal{L}_C(H_i^s, y_i^s) + \lambda_2\, D_k^2(H_i^s, H_i^t) + \lambda_3(e)\, D(H_i^s, H_i^t)_{\mathcal{H}}$$

where $\lambda$ are penalty factors. Because distributions barely match early in training, $\lambda_3$ grows with epoch:

$$\lambda_3(e) = \frac{2}{1 + \exp(-\gamma e / E)} - 1$$

with $e$ the current epoch and $E$ the total epochs. Alignment is applied across **multiple** FC layers (FC2 and FC3) rather than a single layer.

---

## 5. Architecture details (Table II in the paper)

| Network | Layer | Kernel size / stride / filters | Output | Pooling | Padding |
|---|---|---|---|---|---|
| CNN | Conv1D + BN + SELU + MaxPool | 16 / 1×1 / 16 | N×1024×16 | 2×2 | Yes |
| | Conv1D + BN + SELU + MaxPool | 16 / 1×1 / 32 | N×521×32 | 2×2 | Yes |
| | Conv1D + BN + SELU + MaxPool | 8 / 1×1 / 64 | N×256×64 | 2×2 | Yes |
| | Conv1D + BN + SELU + MaxPool | 3 / 1×1 / 128 | N×128×128 | 2×2 | Yes |
| | Conv1D + BN + SELU + Adaptive MaxPool | 3 / 1×1 / 128 | N×4×128 | 32×32 | Yes |
| | Flatten | — | N×512 | — | No |
| Multi-Head Attention | — | — | — | — | — |
| FC | FC1 (Dense + SELU) | 256 | N×256 | — | No |
| | FC2 (Dense + SELU) | 128 | N×128 | — | No |
| | FC3 (Dense + SELU) | 128 | N×128 | — | No |
| Classifier | FC4 + SoftMax | # fault-type neurons | N×10 | — | No |

Multi-head attention parameters (Table I): input N×256, h = 2, query/keys/value N×128, output N×256.

---

## 6. Training setup (implementation details)

| Setting | Value |
|---|---|
| Weight initialization | Xavier |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| $\lambda_1$ (classification) | 1 |
| $\lambda_2$ (global / MK-MMD) | 0.5 |
| $E$ (epochs, used in $\lambda_3$) | 200 |
| Batch size | 128 |
| Repetitions | 10 runs |

---

*Equations and tables are transcribed from the published paper for documentation purposes. Figures remain © 2022 IEEE — see [LICENSE](../LICENSE).*
