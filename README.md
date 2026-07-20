# Lightweight Ciphertext-Only Calibration for Collaborative Learning in Heterogeneous Environments

Post-deployment model calibration performed **entirely under CKKS homomorphic
encryption**. When a deployed classifier misclassifies a sample at a target
site, a peer institution corrects **only the final classifier layer** in the
encrypted domain — no raw data, logits, gradients, or plaintext parameters are
ever exchanged, and no central server is required.

This repository accompanies our work on encrypted post-deployment calibration
and extends the preliminary version presented at **ICNC 2026** *(Privacy-Preserving
Post-Deployment Model Calibration with Fully Homomorphic Encryption)*.

> **Status:** Reference release. The notebooks below reproduce the core
> experiments. Additional utilities, a packaged environment file, and sample
> outputs will be added in upcoming updates.

---

## Key idea

- A querying site (B) encrypts its penultimate feature vector `φ(x)` under its
  own CKKS key and sends the ciphertext to a peer (A).
- Site A runs **ciphertext–plaintext** inference with its classifier head and a
  **Chebyshev polynomial approximation of softmax** (add / multiply / rotate only —
  no exponentials, no division, no bootstrapping).
- Site B decrypts locally, checks `argmax(ŷ) != y`, and — only on a
  misclassification — returns an encrypted one-hot label so A can compute the
  encrypted error `δ = y − ŷ` and apply a bounded last-layer update.
- Secret keys never leave the data-owning site.

The design deliberately stays within a shallow multiplicative depth (≤ 3) so it
runs without CKKS bootstrapping, completing encrypted inference + calibration in
roughly **2–3 s per corrected sample on CPU**.

---

## Repository contents

| Path | Description |
|------|-------------|
| `notebooks/PathMNIST_Heterogeneous.ipynb` | 9-class pathology tiles. Identical-model and different-backbone heterogeneity; B→A encrypted calibration. |
| `notebooks/Pneumonia_MNIST_Heterogeneous.ipynb` | Binary chest X-ray under maximal class skew. Identical-model and different-backbone settings. |
| `notebooks/LLM_MIMIC_Encrypted_Calibration_SingleCell.ipynb` | Clinical notes with frozen **BioClinicalBERT** embeddings. End-to-end training, encrypted calibration, multi-seed + Chebyshev-degree sweep. |

Each notebook is self-contained. The two MedMNIST notebooks have a "Same
Models" section (identical backbones) and a "Different Models" section
(heterogeneous backbones exposing a shared 64-d feature interface).

See [`TUTORIAL.md`](TUTORIAL.md) for a step-by-step walkthrough.

---

## Encryption configuration

The encrypted layer is implemented with [TenSEAL](https://github.com/OpenMined/TenSEAL)
(SEAL backend). Default CKKS context used in the notebooks:

```python
poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
global_scale        = 2 ** 40      # ~200-bit modulus chain, multiplicative depth <= 3
```

- **Chebyshev degree:** swept over `{2, 3, 4, 6}`; degree **4** is used as the
  default (best accuracy/runtime trade-off under the noise budget).
- **Calibration loop:** learning rate `η ∈ {0.05, 0.1}`, `T_max = 20` iterations,
  early stop when `‖δ‖₂ ≤ 1e-3`.
- **Seeds:** `[42, 43, 44, 45, 46]` for multi-seed aggregates.

These are tunable, not intrinsic to the method — larger feature dimensions or
class counts may require adjusting the ring dimension and modulus sizes.

---

## Requirements

See [`requirements.txt`](requirements.txt). Core stack:

- Python 3.9+
- `tenseal` (CKKS)
- `tensorflow` (MedMNIST CNNs + Chebyshev head)
- `torch`, `transformers` (frozen BioClinicalBERT for the MIMIC notebook)
- `medmnist`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Experiments in the paper were run on a CPU node (32-core AMD EPYC 7502,
128 GB RAM). TenSEAL/CKKS execution is CPU-based; a GPU only accelerates the
plaintext CNN / BERT stages.

---

## Data and access

- **PathMNIST / PneumoniaMNIST** download automatically via the `medmnist`
  package.
- **MIMIC-III** is a **credentialed** dataset governed by the PhysioNet Data Use
  Agreement. It is **not** included here and must be obtained from
  [PhysioNet](https://physionet.org/content/mimiciii/). Do **not** commit any
  raw notes, TF-IDF matrices, or cached BioClinicalBERT embeddings derived from
  MIMIC-III to a public repository. The `.gitignore` in this repo is configured
  to exclude those artifacts.

---

## Citation

If you use this code, please cite the ICNC 2026 paper:

```bibtex
@inproceedings{pathan2026encryptedcalibration,
  title     = {Privacy-Preserving Post-Deployment Model Calibration with Fully Homomorphic Encryption},
  author    = {Pathan, Shadman Mahmood Khan and Wang, Qianlong and Takeshita, Jonathan and Imran, Sakan Binte and Shetty, Sachin},
  booktitle = {2026 International Conference on Computing, Networking and Communications (ICNC)},
  year      = {2026},
  pages     = {1--6},
  publisher = {IEEE}
}
```

*(An extended journal version is in preparation; this citation will be updated.)*

---

## Contact

Shadman Mahmood Khan Pathan — `spath004@odu.edu`
Department of Electrical & Computer Engineering, Old Dominion University

## License

See [`LICENSE`](LICENSE). *(Add or confirm the license before making the repo
public — MIT is a common default for academic reference code, but check with
your advisor/co-authors first.)*
