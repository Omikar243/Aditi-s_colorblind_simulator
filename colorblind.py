import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import tempfile
import os
import time

# Check if pillow_heif is installed for HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

def convert_heic_to_pil(heic_file):
    """Convert HEIC image to PIL Image using pillow-heif"""
    if not HEIC_SUPPORT:
        st.error("HEIC support requires the pillow-heif library. Please install it with: pip install pillow-heif")
        return None

    tmp_file_path = None
    try:
        # Create a temporary file
        tmp_file_fd, tmp_file_path = tempfile.mkstemp(suffix='.heic')
        with os.fdopen(tmp_file_fd, 'wb') as tmp_file:
            tmp_file.write(heic_file.getvalue())

        # Open and immediately close the file handle via 'with'
        with Image.open(tmp_file_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image_copy = img.copy()

        return image_copy

    except Exception as e:
        st.error(f"Error converting HEIC image: {str(e)}")
        return None

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            # small pause to ensure all handles are released
            time.sleep(0.1)
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass


def save_pil_as_heic(image):
    """Save PIL Image as HEIC format"""
    if not HEIC_SUPPORT:
        st.error("HEIC support requires the pillow-heif library")
        return None
    
    tmp_file_path = None
    try:
        # Create a temporary file for the HEIC
        tmp_file_fd, tmp_file_path = tempfile.mkstemp(suffix='.heic')
        os.close(tmp_file_fd)  # Close the file descriptor
        
        # Save as HEIC
        heif = pillow_heif.from_pillow(image)
        heif.save(tmp_file_path)
        
        # Read the file contents
        with open(tmp_file_path, 'rb') as f:
            heic_data = f.read()
        
        return heic_data
        
    except Exception as e:
        st.error(f"Error saving HEIC image: {str(e)}")
        return None
    finally:
        # Clean up in a finally block
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                # Small delay to ensure file handle is released
                time.sleep(0.1)
                os.unlink(tmp_file_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")

def apply_protanopia(image):
    """Simulate protanopia (red-green color blindness, absence of red retinal photoreceptors)"""
    # Convert to LMS space
    lms_conversion = np.array([
        [17.8824, 43.5161, 4.11935],
        [3.45565, 27.1554, 3.86714],
        [0.0299566, 0.184309, 1.46709]
    ])
    
    # Simulate protanopia
    protanopia_simulation = np.array([
        [0, 2.02344, -2.52581],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # Convert back to RGB
    rgb_conversion = np.array([
        [0.0809444479, -0.130504409, 0.116721066],
        [-0.0102485335, 0.0540193266, -0.113614708],
        [-0.000365296938, -0.00412161469, 0.693511405]
    ])
    
    # Apply the transformations
    img_array = np.array(image) / 255.0
    height, width, _ = img_array.shape
    
    # Reshape for matrix multiplication
    flat_img = img_array.reshape(-1, 3)
    
    # RGB to LMS
    lms_img = np.dot(flat_img, lms_conversion.T)
    
    # Apply simulation
    sim_lms = np.dot(lms_img, protanopia_simulation.T)
    
    # LMS to RGB
    sim_rgb = np.dot(sim_lms, rgb_conversion.T)
    
    # Reshape and clip values
    sim_img = sim_rgb.reshape(height, width, 3)
    sim_img = np.clip(sim_img, 0, 1) * 255
    
    return Image.fromarray(sim_img.astype(np.uint8))

def apply_deuteranopia(image):
    """Simulate deuteranopia (red-green color blindness, absence of green retinal photoreceptors)"""
    # Convert to LMS space
    lms_conversion = np.array([
        [17.8824, 43.5161, 4.11935],
        [3.45565, 27.1554, 3.86714],
        [0.0299566, 0.184309, 1.46709]
    ])
    
    # Simulate deuteranopia
    deuteranopia_simulation = np.array([
        [1, 0, 0],
        [0.494207, 0, 1.24827],
        [0, 0, 1]
    ])
    
    # Convert back to RGB
    rgb_conversion = np.array([
        [0.0809444479, -0.130504409, 0.116721066],
        [-0.0102485335, 0.0540193266, -0.113614708],
        [-0.000365296938, -0.00412161469, 0.693511405]
    ])
    
    # Apply the transformations
    img_array = np.array(image) / 255.0
    height, width, _ = img_array.shape
    
    # Reshape for matrix multiplication
    flat_img = img_array.reshape(-1, 3)
    
    # RGB to LMS
    lms_img = np.dot(flat_img, lms_conversion.T)
    
    # Apply simulation
    sim_lms = np.dot(lms_img, deuteranopia_simulation.T)
    
    # LMS to RGB
    sim_rgb = np.dot(sim_lms, rgb_conversion.T)
    
    # Reshape and clip values
    sim_img = sim_rgb.reshape(height, width, 3)
    sim_img = np.clip(sim_img, 0, 1) * 255
    
    return Image.fromarray(sim_img.astype(np.uint8))

def apply_tritanopia(image):
    """Simulate tritanopia (blue-yellow color blindness, absence of blue retinal photoreceptors)"""
    # Convert to LMS space
    lms_conversion = np.array([
        [17.8824, 43.5161, 4.11935],
        [3.45565, 27.1554, 3.86714],
        [0.0299566, 0.184309, 1.46709]
    ])
    
    # Simulate tritanopia
    tritanopia_simulation = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-0.395913, 0.801109, 0]
    ])
    
    # Convert back to RGB
    rgb_conversion = np.array([
        [0.0809444479, -0.130504409, 0.116721066],
        [-0.0102485335, 0.0540193266, -0.113614708],
        [-0.000365296938, -0.00412161469, 0.693511405]
    ])
    
    # Apply the transformations
    img_array = np.array(image) / 255.0
    height, width, _ = img_array.shape
    
    # Reshape for matrix multiplication
    flat_img = img_array.reshape(-1, 3)
    
    # RGB to LMS
    lms_img = np.dot(flat_img, lms_conversion.T)
    
    # Apply simulation
    sim_lms = np.dot(lms_img, tritanopia_simulation.T)
    
    # LMS to RGB
    sim_rgb = np.dot(sim_lms, rgb_conversion.T)
    
    # Reshape and clip values
    sim_img = sim_rgb.reshape(height, width, 3)
    sim_img = np.clip(sim_img, 0, 1) * 255
    
    return Image.fromarray(sim_img.astype(np.uint8))

def apply_achromatopsia(image):
    """Simulate achromatopsia (complete color blindness)"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(gray_3channel)

def process_image(image, colorblind_type):
    """Apply the selected colorblind simulation to the image"""
    if colorblind_type == "Protanopia":
        return apply_protanopia(image)
    elif colorblind_type == "Deuteranopia":
        return apply_deuteranopia(image)
    elif colorblind_type == "Tritanopia":
        return apply_tritanopia(image)
    elif colorblind_type == "Achromatopsia":
        return apply_achromatopsia(image)
    else:
        return image  # Normal vision (no change)

def save_image_to_format(image, format_name):
    """Save the image in the specified format"""
    if format_name == "HEIC" and HEIC_SUPPORT:
        return save_pil_as_heic(image)
    else:
        buf = io.BytesIO()
        format_map = {
            "JPEG": "JPEG",
            "PNG": "PNG",
            "WEBP": "WEBP",
            "BMP": "BMP",
            "GIF": "GIF"
        }
        
        image.save(buf, format=format_map.get(format_name, "PNG"))
        return buf.getvalue()

def main():
    st.set_page_config(page_title="Colorblind Simulator", layout="wide")
    
    st.title("Aditi's Colorblind Simulator")
    st.write("""
    Upload an image and see how it might appear to people with different types of color vision deficiency.
    You can download the processed image in various formats.
    
    **Types of Color Blindness:**
    - **Protanopia**: Red-green color blindness (absence of red retinal photoreceptors)
    - **Deuteranopia**: Red-green color blindness (absence of green retinal photoreceptors)
    - **Tritanopia**: Blue-yellow color blindness (absence of blue retinal photoreceptors)
    - **Achromatopsia**: Complete color blindness (monochromacy)
    """)
    
    # Check and notify about HEIC support
    if not HEIC_SUPPORT:
        st.warning("📱 HEIC image support is not enabled. Install the pillow-heif library: `pip install pillow-heif`")
    
    uploaded_file = st.file_uploader(
        "Upload an image (JPEG, PNG, HEIC, etc.)", 
        type=["jpg", "jpeg", "png", "heic"]
    )
    
    colorblind_type = st.selectbox(
        "Select Color Vision Type",
        ["Normal Vision", "Protanopia", "Deuteranopia", "Tritanopia", "Achromatopsia"]
    )
    
    # Available formats for download
    available_formats = ["PNG", "JPEG", "WEBP", "BMP", "GIF"]
    if HEIC_SUPPORT:
        available_formats.append("HEIC")
    
    if uploaded_file is not None:
        try:
            # Handle HEIC format
            if uploaded_file.name.lower().endswith('.heic'):
                if HEIC_SUPPORT:
                    image = convert_heic_to_pil(uploaded_file)
                    if image is None:
                        st.error("Failed to process HEIC image. Please try another format.")
                        st.stop()
                else:
                    st.error("HEIC format is not supported without the pillow-heif library. Please install it or upload a JPEG/PNG image.")
                    st.stop()
            else:
                # Handle standard formats (JPG, PNG)
                image = Image.open(uploaded_file).convert('RGB')
            
            # Process the image
            processed_image = process_image(image, colorblind_type)
            
            # Display original and processed images side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader(f"{colorblind_type} Simulation")
                st.image(processed_image, use_container_width=True)
            
            # Format selection for download
            selected_format = st.selectbox(
                "Select download format",
                available_formats
            )
            
            # Add download button for the processed image
            image_data = save_image_to_format(processed_image, selected_format)
            if image_data:
                file_extension = selected_format.lower()
                
                st.download_button(
                    label=f"Download as {selected_format}",
                    data=image_data,
                    file_name=f"{colorblind_type.lower()}_simulation.{file_extension}",
                    mime=f"image/{file_extension}"
                )
            else:
                st.error(f"Failed to create {selected_format} image for download")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("If you're having issues with HEIC images, make sure you have the pillow-heif library installed.")

if __name__ == "__main__":
    main()
