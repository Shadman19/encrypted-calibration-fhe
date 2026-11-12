# CKKS Encrypted Scoring and Post-Deployment Calibration Experiments

Research notebooks associated with post-deployment calibration and the
**ICNC 2026** paper *Privacy-Preserving Post Deployment Model Calibration with
Fully Homomorphic Encryption*. The paper citation is retained below; repository
implementation scope is stated separately here.

> **Reviewed implementation scope:** in
> `notebooks/LLM_MIMIC_Encrypted_Calibration_SingleCell.ipynb`, the optional
> TenSEAL step encrypts a feature vector, computes ciphertext–plaintext dot
> products, and immediately decrypts the resulting scores. The subsequent
> Chebyshev probability calculation and weight/bias calibration update use
> plaintext NumPy arrays. This notebook does not implement an entirely
> ciphertext-only training or calibration update.

## Implementation boundary in the reviewed MIMIC notebook

- A local TenSEAL context encrypts the selected feature vector for an optional
  scoring/timing step.
- Each encrypted dot product is decrypted before adding the plaintext bias and
  displaying the warmup logits.
- `calibrate_once` computes `cheb_softmax(f @ W + b, ...)`, forms a plaintext
  error, and updates plaintext `W` and `b` arrays.
- The scoring/timing step and the plaintext calibration loop must be reported
  separately. An encrypted scoring demonstration does not establish an
  end-to-end encrypted update or an institution-to-institution deployment.

These statements describe the reviewed code, not a re-evaluation of the paper's
protocol or all other implementations. The PathMNIST and PneumoniaMNIST notebooks
have not been re-audited for their encryption and trust boundaries in this review.
No combined encrypted-inference-and-calibration runtime is established here.

---

## Repository contents

| Path | Description |
|------|-------------|
| `notebooks/PathMNIST_Heterogeneous.ipynb` | Pathology experiment notebook; encryption and adaptation boundaries require separate review. |
| `notebooks/Pneumonia_MNIST_Heterogeneous.ipynb` | Chest X-ray experiment notebook; encryption and adaptation boundaries require separate review. |
| `notebooks/LLM_MIMIC_Encrypted_Calibration_SingleCell.ipynb` | Clinical-note experiment with frozen **BioClinicalBERT** embeddings, optional CKKS dot-product scoring/decryption, and plaintext calibration and evaluation. |

Each notebook is self-contained. The two MedMNIST notebooks have a "Same
Models" section (identical backbones) and a "Different Models" section
(heterogeneous backbones exposing a shared 64-d feature interface).

See [`TUTORIAL.md`](TUTORIAL.md) for a step-by-step walkthrough.

---

## Encryption configuration

The reviewed MIMIC notebook uses [TenSEAL](https://github.com/OpenMined/TenSEAL)
(SEAL backend) for its optional encrypted dot-product step, with this CKKS context:

```python
poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
global_scale        = 2 ** 40
```

Chebyshev degree, calibration steps, learning rate and seeds are configured in
the individual notebook cells. Inspect the selected cell's current settings;
do not assume one common set of defaults across experiments.

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

Distinguish the TenSEAL scoring step from plaintext representation training,
calibration and evaluation when reporting hardware or runtime. This README does
not establish a combined encrypted-calibration performance measurement.

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

Paper: [publisher record](https://doi.org/10.1109/ICNC68183.2026.11416991) | [public PDF](https://www.cs.sjsu.edu/~melody/Pathan_ICNC26_paper.pdf)

```bibtex
@inproceedings{pathan2026encryptedcalibration,
  title     = {Privacy-Preserving Post Deployment Model Calibration with Fully Homomorphic Encryption},
  author    = {Pathan, Shadman Mahmood Khan and Wang, Qianlong and Takeshita, Jonathan and Imran, Sakan Binte and Shetty, Sachin},
  booktitle = {2026 International Conference on Computing, Networking and Communications (ICNC)},
  year      = {2026},
  pages     = {1--6},
  publisher = {IEEE},
  doi       = {10.1109/ICNC68183.2026.11416991},
  url       = {https://doi.org/10.1109/ICNC68183.2026.11416991}
}
```

*(An extended journal version is in preparation; this citation will be updated.)*

---

## Contact

**Shadman Mahmood Khan Pathan** — `spath004@odu.edu`
PhD Student, Department of Electrical & Computer Engineering, Old Dominion University

**Qianlong (Ronnie) Wang, Ph.D.** — `q1wang@odu.edu`
Assistant Professor, Department of Electrical & Computer Engineering, Old Dominion University

## License

See [`LICENSE`](LICENSE) for the MIT license.
