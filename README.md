<div align="center">

<h1>HyperRadar: Learning Structurally Consistent Representations for Multi-View Radar Semantic Segmentation</h1>
<p>
  <a href="https://scholar.google.com/citations?user=BVBJ06QAAAAJ">Ali Zia</a><sup>1,2</sup>,
  <a href="https://scholar.google.com/citations?user=D3AhoccAAAAJ">Muhammad Umer Ramzan</a><sup>3</sup>,
  Abdelwahed Khamis<sup>4</sup>,
  <a href="https://scholar.google.com/citations?user=2A32xVQAAAAJ">Usman Ali</a><sup>3</sup>,
  <a href="https://scholar.google.com/citations?user=ZTuS-yUAAAAJ">Abdul Rehman</a><sup>3</sup>
</p>
<p>
  <sup>1</sup> La Trobe University, Melbourne, Australia<br>
  <sup>2</sup> La Trobe Institute for Sustainable Agriculture &amp; Food (LISAF), Australia<br>
  <sup>3</sup> GIFT University, Gujranwala, Pakistan<br>
  <sup>4</sup> CSIRO, Australia
</p>

<a href="https://arxiv.org/abs/2606.31609">
  <img src="https://img.shields.io/badge/ECCV-2026-1f6feb.svg" alt="ECCV 2026">
</a>
<a href="https://arxiv.org/abs/2606.31609">
  <img src="https://img.shields.io/badge/arXiv-2606.31609-b31b1b.svg" alt="arXiv">
</a>
<a href="https://arxiv.org/pdf/2606.31609">
  <img src="https://img.shields.io/badge/Paper-PDF-blue.svg" alt="PDF">
</a>
<a href="https://ali-zia.me">
  <img src="https://img.shields.io/badge/Author-ali--zia.me-8A2BE2.svg" alt="Author page">
</a>
<a href="LICENSE">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License: Apache 2.0">
</a>

<p>Official code for the ECCV 2026 paper <a href="https://arxiv.org/abs/2606.31609"><i>Learning Structurally Consistent Representations for Multi-View Radar Semantic Segmentation</i></a>.</p>
</div>

## 🧠 Overview

Radar keeps working in rain, fog and darkness, which makes it one of the most dependable sensors for autonomous driving. Segmenting raw radar data is still hard, though. A single object shows up as a scattered group of returns, and the range–Doppler (RD), range–angle (RA) and angle–Doppler (AD) views describe the same scene on different axes with no pixel-to-pixel correspondence between them.

HyperRadar treats both problems as questions of structure:

- **Learnable hypergraphs inside each view.** Every feature location is softly assigned to a small set of hyperedges, so returns that belong together (one pedestrian, one car) exchange information as a group rather than only with their immediate neighbours.
- **Cross-view unbalanced optimal transport.** Features are moved between views with a generalised Sinkhorn solver. Unbalanced OT does not need explicit correspondences and tolerates mass that appears in one view but not another, which is common in radar.
- **Adaptive channel gating and a structural consistency loss.** Cascaded gates fuse what each view has gathered, and a range-profile consistency term keeps the RD and RA predictions in agreement.

HyperRadar reaches **63.8% mIoU on CARRADA** and **83.4% mIoU on RADIal**.

---

## 🏗️ Architecture

<!-- FIGURE: architecture -->

Three temporal encoders (one per radar view) produce latent maps that are projected to a shared width. A learnable hypergraph refines each view on its own. Four cross-view transport plans then carry information into the two output views: RA and AD into RD, and RD and AD into RA. Cascaded channel gates fuse the result before separate RD and RA decoders produce the segmentation masks.

| Component | Paper | Code |
|---|---|---|
| Per-view projection | Eq. 1 | `TransRad.rd_project`, `ra_project`, `ad_project` in [`mvrss/models/TransRadar.py`](mvrss/models/TransRadar.py) |
| Learnable hypergraph | Eq. 2 | `LearnableHypergraph` in [`mvrss/models/hyperradar_blocks.py`](mvrss/models/hyperradar_blocks.py) |
| Unbalanced OT (generalised Sinkhorn) | Eqs. 3–4 | `GeneralizedSinkhorn` |
| Cross-view transfer with learnable β | Eq. 5 | `CrossViewUOT` |
| Adaptive channel gating | Eq. 6 | `ChannelGate` |
| RD / RA decoders | Eq. 7 | `TransRad` decoder branches |
| Training objective | — | `HyperRadarConsistency` in [`mvrss/losses/HyperRadarLoss.py`](mvrss/losses/HyperRadarLoss.py), wired up in `define_loss` ([`mvrss/utils/functions.py`](mvrss/utils/functions.py)) |

---

## 📈 Results

<!-- FIGURE: results -->

| Dataset | mIoU |
|---|---|
| CARRADA | 63.8% |
| RADIal | 83.4% |

See the paper for per-class scores, the RD and RA breakdown, and ablations.

---

## ⚙️ Environment

The code was developed with Python 3.10, PyTorch 1.12.1 and CUDA 11.3 on Linux.

```bash
conda env create -f env.yml
conda activate HyperRadar
pip install -e .
```

---

## 📊 Data

This release trains and evaluates on [CARRADA](https://github.com/valeoai/carrada_dataset). Download and extract it so that the folder layout is:

```
/path/to/datasets/
└── Carrada/
    ├── 2019-09-16-12-52-12/
    ├── ...
    ├── data_seq_ref.json
    └── light_dataset_frame_oriented.json
```

Then tell the code where the data lives and where to write logs (this writes `mvrss/config_files/config.ini`):

```bash
cd mvrss
python -m utils.set_paths --carrada /path/to/datasets --logs /path/to/logs
```

---

## 🚀 Training and testing

All hyperparameters live in [`mvrss/config_files/HyperRadar.json`](mvrss/config_files/HyperRadar.json): 300 epochs, batch size 6, learning rate 1e-4, five input frames, and the `HyperRadar` loss.

```bash
cd mvrss
python -u train.py --cfg ./config_files/HyperRadar.json --cp_store /path/to/checkpoints
```

Training writes a copy of the config into the log folder. Point the test script at it:

```bash
cd mvrss
python -u test.py --cfg /path/to/logs/carrada/TransRadar/TransRadar_0/config.json
```

`train_transrad.sh` and `test_transrad.sh` (run from inside `mvrss/`) wrap the same steps; set `DATA_DIR` at the top of each before running. Add `--unit_test` to `train.py` for a one-batch dry run.

Hypergraph and transport settings (8 hyperedges, ε = 0.05, τ = 0.8, 30 Sinkhorn iterations, a 16×16 transport grid) are constructor defaults of `TransRad`, so they can be changed there without touching the training scripts.

---

## 📁 Repository layout

```
mvrss/
├── config_files/   HyperRadar.json, dataset statistics, path config
├── learners/       training loop, testing and initialisation
├── loaders/        CARRADA dataset and data loaders
├── losses/         HyperRadar consistency loss, soft Dice, coherence
├── models/         HyperRadar (TransRadar.py), its blocks and baseline models
├── utils/          metrics, paths, TensorBoard visualiser
├── train.py
└── test.py
```

---

## 🙏 Acknowledgements

The training and evaluation pipeline is built on [MVRSS](https://github.com/valeoai/MVRSS) (Ouaknine et al., ICCV 2021) and [TransRadar](https://github.com/YahiDar/TransRadar) (Dalbah et al., WACV 2024). The HyperRadar model reuses their encoder and decoder design and keeps their training interface, so configs and scripts from those projects work the same way here. We thank the authors for releasing their code and the CARRADA dataset.

---

## 📖 Citation

If you find this code useful, please cite:

```bibtex
@inproceedings{zia2026hyperradar,
  title     = {Learning Structurally Consistent Representations for Multi-View Radar Semantic Segmentation},
  author    = {Zia, Ali and Ramzan, Muhammad Umer and Khamis, Abdelwahed and Ali, Usman and Rehman, Abdul},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026},
  note      = {arXiv:2606.31609}
}
```

## 📄 License

Released under the [Apache License 2.0](LICENSE), the same licence as the MVRSS code this project builds on.
