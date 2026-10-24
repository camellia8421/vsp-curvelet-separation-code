# Supplemental Materials: Curvelet-Domain Deep Learning for VSP Wavefield Separation

This repository contains the official implementation of the method proposed in the paper, along with all baseline algorithms and data preprocessing scripts for reproducibility.

## 1. Prerequisites

### 1.1 Core Dependencies
- Python ≥ 3.8
- PyTorch ≥ 1.10
- NumPy, SciPy, Pandas, Scikit-learn
- OpenCV-Python
- Matplotlib
- tqdm

### 1.2 Curvelet Transform Library
The curvelet transform is implemented based on **PyCurvelab** (Python wrapper for CurveLab), which is a publicly available standard library.

- Repository: [camellia8421/PyCurvelab](https://github.com/camellia8421/PyCurvelab) (forked from slimgroup/PyCurvelab)
- Underlying library: CurveLab 2.0.2+ ([curvelet.org](http://www.curvelet.org))

#### Installation Steps
1. Install system dependencies: FFTW 3.x, SWIG ≥ 1.3, and a C++ compiler.
2. Download and install CurveLab from the official website, then set environment variables:
   ```bash
   export FDCT=/path/to/your/CurveLab
   export FFTW=/path/to/your/fftw
   ```
3. Clone and install PyCurvelab:
   ```bash
   git clone [https://github.com/camellia8421/PyCurvelab.git](https://github.com/camellia8421/PyCurvelab.git)
   cd PyCurvelab
   python setup.py build install
   ```
4. Verify installation:
   ```python
   import pyct
   help(pyct.fdct2)
   ```

## 2. Repository Structure
```
supplemental_materials/
├── baselines/                  # Baseline methods
│   ├── FCN/                    # Fully Convolutional Network baseline
│   ├── FK/                     # FK fan-filter baseline (physics-driven)
│   └── UNET/                   # Standard U-Net baseline (spatial domain)
├── data_utils.py               # Common data utility functions
├── field_data_process_2.py     # Field VSP data preprocessing script 2
├── field_data_process_shift_1.py # Field VSP data preprocessing script 1 (depth shift)
├── synthetic_data_process.py   # Synthetic dataset preprocessing (time/depth alignment + curvelet transform)
├── proposed_method/            # Proposed curvelet-domain collaborative network
│   ├── cusdataloader.py        # Curvelet-domain dataset loader with subband-wise normalization
│   ├── main.py                 # Training entry
│   ├── test.py                 # Testing & metric evaluation entry
│   ├── utils.py                # Network definition, loss functions, curvelet utilities
│   ├── environment.yml         # Conda environment file
│   └── README.md
└── README.md                   # This file
```

## 3. Usage

### Step 1: Data Preprocessing
1. Place raw synthetic / field VSP data in the `./data/` directory.
2. Run preprocessing to align spatial-temporal dimensions and generate curvelet-domain subbands:
   ```bash
   # For synthetic data
   python synthetic_data_process.py

   # For field data (two-step processing)
   python field_data_process_shift_1.py
   python field_data_process_2.py
   ```
   Outputs will be saved to corresponding directories under `./data/`.

### Step 2: Run Baseline Methods
Navigate to each baseline folder and run the corresponding training / testing script:
- **FK baseline**: physics-driven fan filter, no training required
- **FCN / UNet baselines**: train in spatial domain, following standard end-to-end workflow

### Step 3: Run Proposed Method
```bash
cd proposed_method

# Training
python main.py

# Testing & metric evaluation (R², PSNR, SNR)
python test.py
```
Configuration parameters (data path, batch size, learning rate, etc.) can be modified inside `main.py` and `test.py`.

## 4. Notes
- All curvelet transform parameters (scales, directions, output sizes) are consistent across preprocessing, training and testing to ensure dimension alignment.
- The default curvelet configuration: 3 scales, 8 angles at the finest scale, real-valued wrap-around mode.
- The `cusdataloader.py` supports per-subband normalization, outlier clipping, and automatic retry for corrupted samples.

## 5. Citation
If you use this code in your research, please cite our paper.

## License
The proposed method code is released under the MIT License.
CurveLab and PyCurvelab follow their respective open-source licenses.