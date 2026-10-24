# Supplemental Materials: Curvelet-Domain Deep Learning for VSP Wavefield Separation

Official PyTorch implementation of the paper, including all baselines and preprocessing scripts for full reproducibility.

## Prerequisites

### Core Dependencies
- Python ≥ 3.8
- PyTorch ≥ 1.10
- NumPy, SciPy, Pandas, scikit-learn
- OpenCV-Python, Matplotlib, tqdm

### Curvelet Transform Library
We use **PyCurvelab** (Python wrapper for CurveLab) for the fast discrete curvelet transform.
- Repository: [camellia8421/PyCurvelab](https://github.com/camellia8421/PyCurvelab)
- Backend: CurveLab 2.0.2+ ([curvelet.org](http://www.curvelet.org))

**Installation:**
1. Install system packages: FFTW 3.x, SWIG ≥ 1.3, C++ compiler.
2. Install CurveLab and set environment variables:
   ```bash
   export FDCT=/path/to/CurveLab
   export FFTW=/path/to/fftw
   ```
3. Install PyCurvelab:
   ```bash
   git clone [https://github.com/camellia8421/PyCurvelab.git](https://github.com/camellia8421/PyCurvelab.git)
   cd PyCurvelab
   python setup.py build install
   ```

## Datasets

All datasets are publicly available. Please download from official sources.

### Synthetic VSP Dataset (OpenVSPdata)
- **Source**: [OpenVSPdata Project](https://weiyw16.github.io/en/)
- **Description**: 2D two-component walkaway VSP synthetic dataset, standard benchmark for P/S wave separation.
- **Download**: Google Drive / Baidu Netdisk (code: `hpgc`)
- **Preprocessing**: `synthetic_data_process.py` for spatio-temporal alignment and curvelet transform.
- **Citation**:
  ```bibtex
  @article{wei2022deep,
    title={Deep Learning-Based P- and S-Wave Separation for Multicomponent Vertical Seismic Profiling},
    author={Wei, Y. and Li, Y. E. and Zong, J. and Yang, J. and Fu, H. and Sun, M.},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    volume={60},
    pages={1--16},
    year={2022},
    doi={10.1109/TGRS.2021.3124413}
  }
  ```

### Field VSP Dataset (ISGS CarbonSafe Illinois)
- **Source**: [DOE EDX Platform](https://edx.netl.doe.gov/dataset/illinois-state-geological-survey-isgs-carbonsafe-illinois-macon-county-seismic-data-2d)
- **Publisher**: ISGS, U.S. DOE National Energy Technology Laboratory
- **Description**: Public 2D field seismic dataset for field validation.
- **Preprocessing**: run `field_data_process_shift_1.py` and `field_data_process_2.py` sequentially.

## Repository Structure
```
supplemental_materials/
├── baselines/          # Baseline methods: FCN, FK, UNet
├── data_utils/         # Preprocessing and utility scripts
├── proposed_method/    # Proposed curvelet-domain collaborative network
└── README.md
```

## Usage

### 1. Data Preparation
Place downloaded raw data under `./data/` as described above.

### 2. Preprocessing
```bash
# Synthetic data (alignment + curvelet transform + noise injection)
python synthetic_data_process.py
python synthetic_data_noise.py

# Field data (two-step alignment)
python field_data_process_shift_1.py
python field_data_process_2.py
```
Outputs are saved to corresponding subdirectories under `./data/`.

### 3. Run Baselines
- **FK baseline**: physics-driven filter, no training required.
- **FCN / UNet baselines**: standard end-to-end training in the spatial domain.

### 4. Run Proposed Method
```bash
cd proposed_method
python main.py    # training
python test.py    # evaluation (R², PSNR, SNR)
```
All hyperparameters can be modified directly in `main.py` and `test.py`.

## Notes
- Curvelet transform parameters (scales, directions, sizes) are consistent across preprocessing, training and testing.
- Default configuration: 3 scales, 8 angles at the finest scale, real-valued wrap-around mode.
- Raw data and large checkpoint files should **not** be uploaded to this repository.

## Citation
If you use this code in your research, please cite our paper.

## License
Code of the proposed method is released under the MIT License.
Third-party libraries and datasets follow their respective licenses.
```

---
