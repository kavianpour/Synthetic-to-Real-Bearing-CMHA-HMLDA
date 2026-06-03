# Challenges — why synthetic-to-real bearing diagnosis is hard

This document expands on the real-world problems that motivate CMHA-HMLDA. Each section explains why the problem arises, why it breaks conventional models, and how the proposed method addresses it.

---

## 1. Labeled faulty data is rare in the real world

Unsupervised domain adaptation (UDA) still relies on a large volume of labeled faulty data for training the source model.

**Why it breaks models.** Real recordings are mostly healthy states; faults are rare and may not be captured efficiently; and real-time diagnosis often lacks the domain expertise to label samples. Producing labeled faulty data is the hardest part of real-world deployment, and without it the model's generalization degrades.

**How CMHA-HMLDA responds.** It trains on **labeled synthetic** data (generated from a simple bearing model) as the source and transfers the knowledge to **unlabeled real** data as the target — so no real fault labels are needed.

---

## 2. Global-only alignment leaves local gaps

Most UDA methods are based on the global discrepancy between domains. Distance-based methods like MMD and MK-MMD reduce the distance between the means of feature distributions globally.

**Why it matters.** Global alignment can mix classes near the decision boundary because it ignores the relations between relevant sub-domains, so same-class clusters across domains may not line up.

**How CMHA-HMLDA responds.** It adds **LMMD**, which divides the global distribution into per-class sub-domains and aligns the relevant ones — closing local gaps that global alignment misses.

---

## 3. Sub-domain-only alignment ignores the global picture

Some methods align sub-domains and rely on that to induce global adaptation by forcing local matching.

**Why it matters.** As a drawback, they do not keep global adaptation as a separate regularizer, so the overall source/target match can suffer.

**How CMHA-HMLDA responds.** It keeps **both** a global term (MK-MMD) and a local term (LMMD) as a hybrid loss, so global and local matching reinforce each other rather than trading off.

---

## 4. A single alignment layer is limited

Aligning distributions at only one layer captures only part of the discrepancy.

**How CMHA-HMLDA responds.** It applies alignment across **multiple** FC layers (FC2 and FC3) concurrently, giving a more acceptable distribution match — the ablation shows multi-layer beats single-layer DA.

---

## 5. Distribution shift from conditions and synthetic data

Changing working conditions, environmental noise, and the use of simulated data in training all shift the data distribution, so train and test sets are not drawn from identical distributions, which declines generalization.

**How CMHA-HMLDA responds.** The hybrid multi-layer alignment, combined with a convolutional multi-head attention extractor for more robust features, closes the synthetic→real gap and keeps the diagnosis model reliable. See [`method.md`](method.md).

---

*This document is part of the documentation-only release; the source code is not yet public. See the [Roadmap](../README.md#roadmap).*
