# Canny Edge Detector - Computer Vision Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Step-by-Step Guide](#step-by-step-guide)
7. [Parameters](#parameters)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)
11. [License](#license)

## Project Overview
This project implements the Canny edge detection algorithm, a multi-stage process to detect a wide range of edges in images. The Canny edge detector is widely used in computer vision applications for feature detection and feature extraction.

## Features
- Complete implementation of the Canny edge detection algorithm
- Customizable parameters for different edge detection needs
- Support for various image formats
- Visualization of intermediate processing steps
- Optimized for performance

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.7 or higher
- pip package manager
- Basic understanding of image processing concepts

## Installation
Follow these steps to set up the project:

1. Clone the repository:
```bash
git clone https://github.com/shuvrajit-dey21/computer-vision.git
cd computer-vision
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
To run the Canny edge detector on an image:
```bash
python canny_edge_detector.py --input input_image.jpg --output output_image.png
```

## Step-by-Step Guide
The Canny edge detection process consists of the following steps:

1. **Noise Reduction**
   - The image is smoothed using a Gaussian filter to reduce noise

2. **Gradient Calculation**
   - The intensity gradients of the image are calculated
   - Both magnitude and direction of gradients are computed

3. **Non-Maximum Suppression**
   - Only local maxima in gradient directions are kept
   - This thins the edges

4. **Double Thresholding**
   - Potential edges are determined by two threshold values
   - Strong edges, weak edges, and non-edges are classified

5. **Edge Tracking by Hysteresis**
   - Weak edges are either kept or discarded based on connectivity
   - Only edges connected to strong edges are preserved

## Parameters
The script accepts the following parameters:
- `--input`: Path to input image (required)
- `--output`: Path to save output image (required)
- `--low_threshold`: Low threshold for hysteresis (default: 50)
- `--high_threshold`: High threshold for hysteresis (default: 150)
- `--sigma`: Sigma value for Gaussian blur (default: 1.4)
- `--kernel_size`: Size of Gaussian kernel (default: 5)

## Examples
1. Basic edge detection:
```bash
python canny_edge_detector.py --input sample.jpg --output edges.png
```

2. Custom threshold values:
```bash
python canny_edge_detector.py --input sample.jpg --output edges.png --low_threshold 30 --high_threshold 100
```

3. Different blur settings:
```bash
python canny_edge_detector.py --input sample.jpg --output edges.png --sigma 2.0 --kernel_size 7
```

## Troubleshooting
- **Image not found**: Ensure the input path is correct
- **Blurry output**: Try adjusting the sigma and kernel_size parameters
- **Too many/too few edges**: Adjust the threshold values
- **Memory issues**: Reduce image size before processing

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
