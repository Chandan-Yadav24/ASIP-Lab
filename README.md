# ASIP Lab – Advanced Signal & Image Processing Interactive Learning Platform

An interactive, browser-based educational platform for learning signal processing and image processing concepts. Built with [Streamlit](https://streamlit.io/), this project provides hands-on demonstrations of fundamental and advanced DSP and computer vision techniques.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B)
![OpenCV](https://img.shields.io/badge/OpenCV-Latest-5C3EE8)
![NumPy](https://img.shields.io/badge/NumPy-Latest-013243)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Topics Covered](#topics-covered)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **50+ Interactive Demonstrations** – Explore signal and image processing concepts with real-time visualizations
- **Real-time Parameter Control** – Adjust parameters and see immediate results
- **Educational Focus** – Clear explanations paired with code examples
- **No Installation Required** – Run directly in your browser after setup
- **Comprehensive Coverage** – From basics to advanced techniques
- **Mathematical Visualizations** – Understand complex concepts through interactive plots

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Chandan-Yadav24/ASIP-Lab.git
   cd ASIP-Lab
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the Streamlit application:
```bash
streamlit run main.py
```

The application will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
ASIP-Lab/
├── main.py                           # Main Streamlit application entry point
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── demos/                            # Core demonstration modules
│   ├── __init__.py
│   ├── dsp_utils.py                 # Utility functions for DSP operations
│   ├── signals_demo.py              # Basic signals
│   ├── wav_demo.py                  # WAV file processing
│   ├── noise_demo.py                # Noise generation and analysis
│   ├── dft_demo.py                  # Discrete Fourier Transform
│   ├── fft_mechanics_demo.py        # FFT mechanics
│   ├── freq_domain_demo.py          # Frequency domain analysis
│   ├── correlation_demo.py          # Correlation operations
│   ├── autocorrelation_demo.py      # Autocorrelation
│   ├── convolution_filtering_demo.py # Convolution and filtering
│   ├── smoothing_lpf_demo.py        # Low-pass filtering
│   ├── sharpening_hpf_demo.py       # High-pass filtering
│   ├── image_processing_intro_demo.py # Image processing basics
│   ├── intensity_transform_demo.py  # Intensity transformations
│   ├── thresholding_demo.py         # Image thresholding
│   ├── histogram_equalization_demo.py # Histogram operations
│   ├── edge_detection/              # Edge detection techniques
│   │   ├── canny_edge_demo.py
│   │   ├── sobel_edge_demo.py
│   │   ├── laplacian_demo.py
│   │   └── ...
│   ├── morphology/                  # Morphological operations
│   │   ├── erosion_demo.py
│   │   ├── dilation_demo.py
│   │   ├── opening_closing_demo.py
│   │   └── ...
│   ├── feature_extraction/          # Feature detection and extraction
│   │   ├── harris_corner_demo.py
│   │   ├── sift_demo.py
│   │   ├── hog_demo.py
│   │   ├── haar_features_demo.py
│   │   └── ...
│   └── segmentation/                # Image segmentation
│       ├── region_growing_demo.py
│       ├── watershed_demo.py
│       ├── grabcut_demo.py
│       └── ...
└── pages/                           # Multi-page app structure (optional)
```

## 🎓 Topics Covered

### Signal Processing
- **Fundamentals:** Periodic signals, signal objects, sampling
- **Frequency Domain:** DFT, FFT, spectral analysis, frequency response
- **Noise & Filtering:** Gaussian noise, Brownian motion, pink noise, correlation
- **Filtering Techniques:** Low-pass, high-pass, convolution, morphological filtering
- **Advanced:** Integrated spectrums, overlapping fields, Hough transforms

### Image Processing
- **Basics:** Image types, formats, intensity transforms, gamma correction
- **Enhancement:** Contrast stretching, histogram equalization, sharpening
- **Edge Detection:** Canny, Sobel, Roberts, Laplacian, LoG, DoG
- **Morphological Operations:** Erosion, dilation, opening, closing, hit-or-miss, skeletonization
- **Feature Extraction:** Harris corners, SIFT, HOG, Haar features, boundary features
- **Image Pyramids:** Gaussian and Laplacian pyramids, multi-scale analysis
- **Segmentation:** Region growing, region splitting/merging, watershed, active contours, GrabCut
- **Advanced:** PCA, morphological snakes, blob detection

## 📦 Requirements

All dependencies are listed in `requirements.txt`:

- **streamlit** – Web app framework
- **opencv-python** – Computer vision library
- **numpy** – Numerical computing
- **scipy** – Scientific computing
- **matplotlib** – Visualization
- **pillow** – Image processing
- **pandas** – Data manipulation
- **scikit-image** – Image processing algorithms

Install all at once:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Navigating the Platform

1. **Main Dashboard** – Start page with category selection
2. **Select a Unit** – Choose from signal processing or image processing topics
3. **Interactive Controls** – Adjust parameters using sliders and input fields
4. **Real-time Visualization** – See results update instantly
5. **Learn & Experiment** – Read descriptions and experiment with different values

### Example Workflows

- **Understanding FFT:** Start with basic signals, then explore FFT mechanics
- **Edge Detection:** Learn intensity transforms → gradients → edge detectors
- **Image Segmentation:** Understand thresholding → region-based methods → active contours
- **Feature Extraction:** Master corner detection → SIFT → HOG features

## 💡 Tips for Learning

- Start with fundamentals (signals, basic image operations)
- Progress to frequency domain analysis
- Explore edge detection and feature extraction next
- Finally, try advanced segmentation and morphological techniques
- Adjust parameters to see how they affect results
- Combine multiple techniques to understand their interactions

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution
- New demo implementations
- Improved visualizations
- Better documentation
- Bug fixes
- Performance optimizations
- Additional examples

## 📝 License

This project is licensed under the MIT License – see the LICENSE file for details.

## 👨‍💻 Author

Created as an educational resource for understanding signal and image processing concepts.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Computer vision with [OpenCV](https://opencv.org/)
- Scientific computing with [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/)
- Visualization with [Matplotlib](https://matplotlib.org/)

## 📧 Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Submit a pull request
- Reach out through project discussions

---

**Happy Learning! Explore, experiment, and master signal and image processing concepts.** 🎯
