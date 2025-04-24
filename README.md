# Canny Edge Detection Algorithm Implementation

## Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage
```python
python canny_edge_detector.py
```
1. Click 'Choose Image' to select an image
2. Click 'Process' to run edge detection
3. Use 'Reset' to return to default image
4. Click ℹ️ button for algorithm details

## Features

### Image Handling
- Supports various image formats (JPG, JPEG, PNG, BMP, GIF)
- Automatically converts non-RGB images to RGB format
- Preserves image quality during processing

### Visualization
- Optimized display dimensions (400x450 pixels) for better visibility
- Maintains aspect ratio during image scaling
- Centers images in the display area
- Smooth scrolling for larger content

### Image Size Handling
- Large images are automatically scaled down while preserving aspect ratio
- Small images maintain original dimensions
- No image cropping or distortion
- Gray background for partially filled frames

### Error Handling
- Graceful handling of invalid image formats
- Clear error messages for failed operations
- Status updates during processing

### User Interface
- Clean and intuitive layout
- Responsive controls
- Real-time progress tracking
- Informative status messages

## Features

### Core Features
- Complete manual implementation of the Canny Edge Detection algorithm
- User-friendly GUI built with Tkinter
- Support for various image formats (PNG, JPG, JPEG, BMP, GIF, TIFF)
- Real-time image processing and display
- Side-by-side view of original and processed images

### UI Features
- Modern, responsive interface with smooth animations
- Interactive algorithm information window with step-by-step explanations
- Progress bar with percentage indicator during processing
- Smooth scrolling and fade effects
- Hover effects on interactive elements
- Reset functionality to return to default image

## Requirements

- Python 3.7 or higher
- NumPy (>= 1.21.0)
- Pillow (>= 9.0.0)

## Installation

1. Clone this repository or download the source code
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python canny_edge_detector.py
```

2. The application will start with a default test image (black and white squares)
3. Use the following controls:
   - Click "Choose Image" to select your own image
   - Click "Process" to apply the Canny Edge Detection algorithm
   - Click "Reset" to return to the default test image
   - Click "ℹ️ Algorithm Info" to learn about the algorithm steps

## Project Structure

- `canny_edge_detector.py`: Main application file containing:
  - `CannyEdgeDetector`: Main application class with UI implementation
  - `AlgorithmInfoWindow`: Interactive information window class
- `requirements.txt`: Project dependencies

## Algorithm Implementation

The Canny Edge Detection algorithm is implemented in the following steps:

1. **Grayscale Conversion**
   - Converts RGB to grayscale using weighted sum:
   - Formula: gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

2. **Gaussian Blur**
   - Reduces noise while preserving edges
   - Uses 5x5 Gaussian kernel with formula: G(x,y) = (1/2πσ²)e^(-(x²+y²)/2σ²)

3. **Gradient Calculation**
   - Applies Sobel operators in x and y directions
   - Calculates gradient magnitude: √(Gx² + Gy²)
   - Determines gradient direction: θ = arctan(Gy/Gx)

4. **Non-Maximum Suppression**
   - Thins edges by suppressing non-maximum values
   - Rounds gradient direction to nearest 45°
   - Compares with pixels in gradient direction

5. **Double Thresholding**
   - Identifies strong edges (high threshold: 0.15 * max)
   - Identifies weak edges (low threshold: 0.05 * max)
   - Categorizes pixels as strong, weak, or non-edges

6. **Edge Tracking by Hysteresis**
   - Starts with strong edges
   - Recursively adds connected weak edges
   - Removes isolated weak edges

## UI Design

The application features a modern, responsive interface with:

- Centered title with custom styling
- Organized button layout with consistent spacing
- Progress tracking during image processing
- Smooth animations and transitions
- Interactive hover effects
- Scrollable content for larger images
- Side-by-side image comparison

## Implementation Details

- All image processing operations are implemented manually without using OpenCV
- Uses NumPy for efficient mathematical operations
- Uses Pillow (PIL) for basic image loading and display
- Implements custom UI components with Tkinter
- Features smooth animations and transitions
- Includes detailed algorithm explanations

## License

This project is open-source and available under the MIT License.