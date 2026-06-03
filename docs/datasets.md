# Dataset — Synthetic & real CWRU bearing data, and the generation model

This document describes the data used in the paper, the synthetic-to-real setup, the classes, the synthetic-data generation model, and how to obtain the data.

---

## 1. The synthetic-to-real setup

| Property | Detail |
|---|---|
| Benchmark | Case Western Reserve University (CWRU) bearing dataset |
| Source domain | Labeled **synthetic** CWRU data (generated from a simple bearing model) |
| Target domain | Unlabeled **real** CWRU data |
| Sensors | Drive-end and fan-end bearings |
| Sample rate | 12 kHz |

The CWRU dataset is one of the most widely used benchmarks for unsupervised bearing fault diagnosis.

---

## 2. Health classes and sampling

Four health states are used: **Healthy, Inner-race fault, Ball fault, Outer-race fault**.

- Per class: **1200 segments × 4096 points** → **4800 samples** across all health states per domain.
- Healthy data is the same in source and target domains, so only half of it is taken in each domain.

---

## 3. Generating synthetic faulty data

The assumption is access to real **healthy** vibration data. Synthetic data $\epsilon(t)$ is produced as:

$$\epsilon(t) = \sum_{i=-\infty}^{\infty} A_i\, s(t - iT) + \beta\, n(t)$$

where $n$ is healthy vibration data, $\beta$ is uniformly distributed in [0.25, 2], and $s$ is modeled as a single impact via a Hann window with a 5 % duty period relative to $T$. The amplitude term $A_i$ is a sum of cosines:

$$A_i = \gamma \sum_{k=0}^{K} \alpha_k \cos(iTk\,2\pi/Q)$$

where $K$ indicates side-bands and the amplitude set is $\alpha = [1, 0.76, 0.38, 0.11, 0.05]$. (A normally-distributed factor with unit mean and standard deviation 0.1 is also applied.)

---

## 4. Obtaining the data

The real **CWRU bearing dataset** is publicly distributed by Case Western Reserve University and widely mirrored. The **synthetic** data is produced from real healthy data using the model above. Neither is redistributed in this repository. To reproduce:

1. Obtain the real CWRU drive-end and fan-end data at 12 kHz for the four health states.
2. Segment into 1200 × 4096-point windows per class (4800 per domain); share healthy data across domains.
3. Generate synthetic faulty data from real healthy data using the $\epsilon(t)$ model (source domain).
4. Treat real data as the unlabeled target; produce target pseudo-labels during training.

> Always check and comply with the dataset's usage terms.

---

*This document is part of the documentation-only release; synthetic-data generation scripts will accompany the code when released. See the [Roadmap](../README.md#roadmap).*
