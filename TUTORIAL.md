# Tutorial: Running Encrypted Post-Deployment Calibration

This walkthrough explains how the encrypted calibration pipeline works and how
to run each notebook. It assumes basic familiarity with Python notebooks and
homomorphic encryption at a conceptual level.

---

## 1. The pipeline in four stages

Every experiment follows the same four stages, regardless of dataset or model:

1. **Local non-IID training.** Each site (A and B) trains its own model on a
   class-skewed partition. The classifier head uses a **Chebyshev polynomial
   approximation of softmax** instead of the standard exponential softmax, so
   the forward pass is homomorphic-encryption friendly (additions and
   multiplications only).

2. **Encrypted inference.** Site B extracts the penultimate feature
   `φ(x) ∈ R^d`, encrypts it under its own CKKS key, and sends the ciphertext to
   the peer. The peer computes encrypted logits `z = Wᵀ·φ(x) + b` and the
   Chebyshev softmax surrogate — all in ciphertext.

3. **Local misclassification check.** The encrypted scores are returned to Site
   B, which decrypts locally and tests `argmax(ŷ) ≠ y`. Decryption happens only
   at the data-owning site; the peer never sees plaintext.

4. **Ciphertext-only last-layer update.** If (and only if) the prediction was
   wrong, Site B returns an encrypted one-hot label. The peer computes the
   encrypted error `δ = y − ŷ` and updates **only the final layer**:

   ```
   W_i ← W_i + η · δ_i · φ(x)
   b_i ← b_i + η · δ_i          for i = 1..C
   ```

   Iterations stop when `‖δ‖₂ ≤ ε` or after `T_max` steps. No gradients,
   logits, or raw inputs are ever exchanged in plaintext.

---

## 2. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

If `tenseal` fails to build from source, install a prebuilt wheel for your
platform/Python version (see the TenSEAL README). TenSEAL requires a working
C++ toolchain when compiling.

---

## 3. PathMNIST (9-class vision)

Open `notebooks/PathMNIST_Heterogeneous.ipynb`.

- **Cell "Same Models":** trains identical Chebyshev-head CNNs on non-IID splits
  (Site A → classes 0–4, Site B → classes 5–8). It reports per-site validation
  accuracy and AUROC, then performs a **B→A** encrypted calibration pass on a
  batch of misclassified features and reports the pre/post transfer accuracy and
  AUROC. Expected: a clear jump in B→A accuracy and AUROC after calibration, and
  a single misclassified tile whose true-class probability rises and whose
  `argmax` flips.
- **Cell "Different Models":** repeats the experiment with **different CNN
  backbones** at A and B that still expose a shared 64-d penultimate feature.
  Demonstrates that the encrypted head update is robust to architectural
  mismatch.

Data downloads automatically through the `medmnist` package on first run.

Each run produces "before vs. after" probability bars for a representative
misclassified sample and a per-step calibration-loss curve (monotone descent
under the 20-step budget).

---

## 4. PneumoniaMNIST (binary vision)

Open `notebooks/Pneumonia_MNIST_Heterogeneous.ipynb`.

Same structure ("Same Model" / "Different Models"), but with **maximal binary
skew**: Site A enriched for *normal*, Site B enriched for *pneumonia*.
Thresholds are chosen with Youden's *J* on each local validation split. This is
the hardest generalization case — cross-site transfer before calibration is
poor, and a single encrypted last-layer update flips the decision in place.

---

## 5. MIMIC-III clinical notes (text, frozen BioClinicalBERT)

Open `notebooks/LLM_MIMIC_Encrypted_Calibration_SingleCell.ipynb`.

> **Prerequisite:** You must have credentialed access to MIMIC-III and the
> relevant note tables locally. This notebook does **not** ship any clinical
> data. Point the data-loading section at your local, de-identified files.

This is a single self-contained cell that runs the full experiment:

1. Builds the clinical representation (frozen **BioClinicalBERT** embeddings, or
   the TF-IDF variant) with sites split by label skew (A → survivors,
   B → non-survivors).
2. Trains a lightweight classifier head with the Chebyshev surrogate.
3. Runs encrypted last-layer calibration on misclassified target-site notes.
4. Aggregates results across seeds `[42, 43, 44, 45, 46]` and performs a
   **Chebyshev degree sweep** over `{2, 3, 4, 6}`, writing summary CSVs and
   figures.

Only the final classification head is updated under encryption; the transformer
backbone stays frozen. This is a **feasibility result** — encrypted calibration
runs with add/multiply/rotate-only CKKS operations while preserving
discrimination and improving probability reliability.

---

## 6. What to look for

Across all three notebooks, a successful encrypted calibration run shows:

- **Monotone `L2` loss descent** over the calibration steps (numerical stability
  under ciphertext noise).
- **True-class probability increases** and, where applicable, the `argmax` flips
  to the correct label.
- **Preserved ranking metrics** (AUROC / AUPRC) after the update, since only a
  small localized change is applied to the last layer.
- **~2–3 s per corrected sample** end-to-end on CPU.

---

## 7. Tuning knobs

| Knob | Where | Effect |
|------|-------|--------|
| `poly_modulus_degree`, `coeff_mod_bit_sizes`, `global_scale` | CKKS context | Noise budget vs. runtime / ciphertext size |
| Chebyshev degree | head definition / degree sweep | Approximation quality vs. multiplicative depth |
| `η` (learning rate) | calibration loop | Update strength vs. stability |
| `T_max`, `ε` | stopping rule | Iteration budget vs. convergence |
| `primary_ratio` | data split | Severity of non-IID skew |

Start from the defaults in the README; re-tune per dataset/architecture as
needed.
