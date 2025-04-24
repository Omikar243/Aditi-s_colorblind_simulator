# Colorblind Simulator

A Streamlit web application that simulates how images appear to people with different types of color vision deficiency (color blindness).

## Features

- Upload images in various formats (JPEG, PNG, HEIC, etc.)
- Simulate four different types of color blindness:
  - **Protanopia**: Red-green color blindness (absence of red retinal photoreceptors)
  - **Deuteranopia**: Red-green color blindness (absence of green retinal photoreceptors)
  - **Tritanopia**: Blue-yellow color blindness (absence of blue retinal photoreceptors)
  - **Achromatopsia**: Complete color blindness (monochromacy)
- Side-by-side comparison of original and simulated images
- Download processed images in multiple formats (PNG, JPEG, WEBP, BMP, GIF, HEIC)

## Requirements

- Python 3.8+
- Streamlit
- NumPy
- OpenCV (cv2)
- Pillow (PIL)
- pillow-heif (optional, for HEIC support)

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## HEIC Support

For HEIC image support (common for iPhone photos), install the pillow-heif library:
```bash
pip install pillow-heif
```

## Usage

1. Upload an image using the file uploader
2. Select a color vision type from the dropdown menu
3. View the original and simulated images side by side
4. Select a format and download the processed image

## How It Works

The app uses color transformation matrices to simulate how images would appear to people with different types of color blindness. It converts RGB images to the LMS color space (which represents the response of the three types of cones in the human eye), modifies the values according to the selected color vision deficiency, and then converts back to RGB.

## Deployment

This app can be deployed on Streamlit Sharing or any other platform that supports Streamlit applications.
