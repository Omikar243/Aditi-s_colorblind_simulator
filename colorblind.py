import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import tempfile
import os
import time
import matplotlib.pyplot as plt
import threading
import queue

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
        st.error(
            "HEIC support requires the pillow-heif library. "
            "Please install it with: pip install pillow-heif"
        )
        return None

    tmp_file_path = None
    try:
        # Create a temporary file
        tmp_fd, tmp_file_path = tempfile.mkstemp(suffix=".heic")
        with os.fdopen(tmp_fd, "wb") as tmp:
            tmp.write(heic_file.getvalue())

        # Open/close with context manager to release the lock
        with Image.open(tmp_file_path) as img:
            img = img.convert("RGB")
            return img.copy()

    except Exception as e:
        st.error(f"Error converting HEIC image: {e}")
        return None

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            time.sleep(0.05)
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass


def save_pil_as_heic(image: Image.Image) -> bytes:
    """Save PIL Image as HEIC format"""
    if not HEIC_SUPPORT:
        st.error("HEIC support requires the pillow-heif library")
        return None

    tmp_file_path = None
    try:
        tmp_fd, tmp_file_path = tempfile.mkstemp(suffix=".heic")
        os.close(tmp_fd)

        heif = pillow_heif.from_pillow(image)
        heif.save(tmp_file_path)

        with open(tmp_file_path, "rb") as f:
            return f.read()

    except Exception as e:
        st.error(f"Error saving HEIC image: {e}")
        return None

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            time.sleep(0.05)
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass


# Simulation matrices for different types of color blindness
SIMULATION_MATRICES = {
    "protanopia": np.array([
        [0.567, 0.433, 0.0],
        [0.558, 0.442, 0.0],
        [0.0,   0.242, 0.758]
    ]),
    "deuteranopia": np.array([
        [0.625, 0.375, 0.0],
        [0.70,  0.30,  0.0],
        [0.0,   0.30,  0.70]
    ]),
    "tritanopia": np.array([
        [0.95,  0.05,  0.0],
        [0.0,   0.433, 0.567],
        [0.0,   0.475, 0.525]
    ])
}


def apply_colorblind_simulation(img: np.ndarray, deficiency_type: str, severity: float) -> np.ndarray:
    """Apply the 3×3 simulation matrix to an RGB image with a given severity."""
    if deficiency_type.lower() == "achromatopsia":
        # For achromatopsia, convert to grayscale with severity interpolation
        gray = np.dot(img, [0.299, 0.587, 0.114])
        gray_rgb = np.stack([gray, gray, gray], axis=-1)
        return (img * (1 - severity) + gray_rgb * severity).clip(0, 255)
    
    matrix = SIMULATION_MATRICES.get(deficiency_type)
    if matrix is None:
        return img
    # Blend between original and fully simulated
    simulated = img.dot(matrix.T)
    return (img * (1 - severity) + simulated * severity).clip(0, 255)


def apply_protanopia(image: Image.Image, severity: float = 1.0) -> Image.Image:
    """
    Simulate protanopia (red-blind) using the standard protanopia matrix:
      R' = 0.567R + 0.433G + 0B
      G' = 0.558R + 0.442G + 0B
      B' = 0   R + 0.242G + 0.758B
    """
    M = np.array([
        [0.567, 0.433, 0.0],
        [0.558, 0.442, 0.0],
        [0.0,   0.242, 0.758]
    ])
    
    # Interpolate with identity matrix based on severity
    I = np.eye(3)
    transform_matrix = (1 - severity) * I + severity * M
    
    arr = np.array(image, dtype=np.float32)
    flat = arr.reshape(-1, 3)
    sim = flat.dot(transform_matrix.T)
    sim = np.clip(sim, 0, 255).astype(np.uint8)
    return Image.fromarray(sim.reshape(arr.shape))


def apply_deuteranopia(image: Image.Image, severity: float = 1.0) -> Image.Image:
    """
    Simulate deuteranopia (green-blind) using the standard deuteranopia matrix:
      R' = 0.625R + 0.375G + 0B
      G' = 0.70 R + 0.30 G + 0B
      B' = 0   R + 0.30 G + 0.70 B
    """
    M = np.array([
        [0.625, 0.375, 0.0],
        [0.70,  0.30,  0.0],
        [0.0,   0.30,  0.70]
    ])
    
    # Interpolate with identity matrix based on severity
    I = np.eye(3)
    transform_matrix = (1 - severity) * I + severity * M
    
    arr = np.array(image, dtype=np.float32)
    flat = arr.reshape(-1, 3)
    sim = flat.dot(transform_matrix.T)
    sim = np.clip(sim, 0, 255).astype(np.uint8)
    return Image.fromarray(sim.reshape(arr.shape))


def apply_tritanopia(image: Image.Image, severity: float = 1.0) -> Image.Image:
    """
    Simulate tritanopia (blue-blind) using the standard tritanopia matrix:
      R' = 0.95 R + 0.05 G + 0B
      G' = 0   R + 0.433 G + 0.567 B
      B' = 0   R + 0.475 G + 0.525 B
    """
    M = np.array([
        [0.95,  0.05,  0.0],
        [0.0,   0.433, 0.567],
        [0.0,   0.475, 0.525]
    ])
    
    # Interpolate with identity matrix based on severity
    I = np.eye(3)
    transform_matrix = (1 - severity) * I + severity * M
    
    arr = np.array(image, dtype=np.float32)
    flat = arr.reshape(-1, 3)
    sim = flat.dot(transform_matrix.T)
    sim = np.clip(sim, 0, 255).astype(np.uint8)
    return Image.fromarray(sim.reshape(arr.shape))


def apply_achromatopsia(image: Image.Image, severity: float = 1.0) -> Image.Image:
    """Simulate complete color blindness by converting to grayscale."""
    if severity == 1.0:
        return image.convert("L").convert("RGB")
    else:
        # Interpolate between original and grayscale
        original = np.array(image, dtype=np.float32)
        gray = np.array(image.convert("L").convert("RGB"), dtype=np.float32)
        result = (1 - severity) * original + severity * gray
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def process_image(image: Image.Image, mode: str, severity: float = 1.0) -> Image.Image:
    """Process image with specified color vision deficiency and severity"""
    if mode == "Protanopia":
        return apply_protanopia(image, severity)
    elif mode == "Deuteranopia":
        return apply_deuteranopia(image, severity)
    elif mode == "Tritanopia":
        return apply_tritanopia(image, severity)
    elif mode == "Achromatopsia":
        return apply_achromatopsia(image, severity)
    else:
        return image


def process_frame_cv2(frame: np.ndarray, mode: str, severity: float = 1.0) -> np.ndarray:
    """Process a single frame from OpenCV (BGR format)"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # No simulation for normal vision
    if mode == "Normal Vision":
        return frame
    # Determine deficiency key exactly as in our matrices
    deficiency_type = mode.lower().replace("anomaly", "opia")
    # Apply simulation
    processed_rgb = apply_colorblind_simulation(frame_rgb.astype(np.float32), deficiency_type, severity)
    # Convert back to BGR for OpenCV
    processed_bgr = cv2.cvtColor(processed_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return processed_bgr


def webcam_processor(frame_queue: queue.Queue, processed_queue: queue.Queue,
                     mode: str, severity: float, stop_event: threading.Event):
    """Process frames from a webcam in a background thread."""
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        processed = process_frame_cv2(frame, mode, severity)
        processed_queue.put(processed)


def save_image_to_format(image: Image.Image, fmt: str) -> bytes:
    if fmt == "HEIC" and HEIC_SUPPORT:
        return save_pil_as_heic(image)
    buf = io.BytesIO()
    fmt_map = {"JPEG":"JPEG","PNG":"PNG","WEBP":"WEBP","BMP":"BMP","GIF":"GIF"}
    image.save(buf, fmt_map.get(fmt, "PNG"))
    return buf.getvalue()


def create_color_wheel(size=400) -> Image.Image:
    """Generate a demo color wheel for when no upload is provided."""
    img = np.ones((size, size, 3), dtype=np.uint8)*255
    c = size//2
    r = c - 10
    for y in range(size):
        for x in range(size):
            dx, dy = x-c, y-c
            d = np.hypot(dx, dy)
            if d <= r:
                angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
                hue = angle / 360.0
                rgb = plt.cm.hsv(hue)[:3]
                sat = d/r
                rgb = tuple(int(255*(sat*c_ + (1-sat))) for c_ in rgb)
                img[y, x] = rgb
    return Image.fromarray(img)


def process_video_file(video_path: str, output_path: str, mode: str, severity: float):
    """Process uploaded video file with color blindness simulation"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    progress_bar = st.progress(0)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame = process_frame_cv2(frame, mode, severity)
        out.write(processed_frame)
        
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
    
    cap.release()
    out.release()
    progress_bar.empty()


def process_video_file_all_types(video_path: str, output_paths: dict):
    """Process uploaded video file with all color blindness simulation types"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writers
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writers = {}
    for mode, path in output_paths.items():
        writers[mode] = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    progress_bar = st.progress(0)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame for each type
        for mode, writer in writers.items():
            if mode == "Normal":
                processed_frame = frame
            else:
                processed_frame = process_frame_cv2(frame, mode, 1.0)
            writer.write(processed_frame)
        
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
    
    cap.release()
    for writer in writers.values():
        writer.release()
    progress_bar.empty()


def main():
    st.set_page_config(page_title="Enhanced Colorblind Simulator", layout="wide")
    st.title("Aditi's Enhanced Colorblind Simulator")
    st.write("""
    Upload an image or video, use your webcam, and see how content appears under different color vision deficiencies.
    Now with adjustable severity levels and support for anomalous trichromacy!
    """)

    if not HEIC_SUPPORT:
        st.warning("Install `pillow-heif` for HEIC support: `pip install pillow-heif`")

    # Main dashboard controls
    st.header("Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Mode selection with new options
        vision_types = [
            "Normal Vision", "Protanopia", "Deuteranopia", "Tritanopia", "Achromatopsia"
        ]
        mode = st.selectbox("Select vision type", vision_types)
    
    with col2:
        # Severity slider
        if mode != "Normal Vision":
            severity = st.slider(
                "Severity Level (%)", 
                min_value=0, 
                max_value=100, 
                value=100,
                help="0% = Normal vision, 100% = Full deficiency"
            ) / 100.0
        else:
            severity = 0.0
    
    with col3:
        # Input method selection
        input_method = st.selectbox(
            "Choose input method:",
            ["Upload Image", "Upload Video", "Live Webcam"]
        )
    
    all_view = st.checkbox("Show all types side by side")

    formats = ["PNG","JPEG","WEBP","BMP","GIF"]
    if HEIC_SUPPORT: formats.append("HEIC")

    # Handle different input methods
    if input_method == "Upload Image":
        uploaded = st.file_uploader("Upload image (JPEG, PNG, HEIC…)", type=["jpg","jpeg","png","heic"])
        
        # load or demo
        if uploaded:
            if uploaded.name.lower().endswith(".heic"):
                img = convert_heic_to_pil(uploaded) or create_color_wheel()
            else:
                img = Image.open(uploaded).convert("RGB")
        else:
            st.info("No upload—showing demo color wheel")
            img = create_color_wheel()

        if all_view:
            st.subheader("All Vision Types")
            imgs = {
                "Normal": img,
                "Protanopia": process_image(img, "Protanopia", 1.0),
                "Deuteranopia": process_image(img, "Deuteranopia", 1.0),
                "Tritanopia": process_image(img, "Tritanopia", 1.0),
                "Achromatopsia": process_image(img, "Achromatopsia", 1.0)
            }
            
            # Display in grid
            cols = st.columns(3)
            for i, (title, im) in enumerate(imgs.items()):
                with cols[i % 3]:
                    st.write(title)
                    st.image(im, width="stretch")
            
            st.markdown("---")
            for title, im in imgs.items():
                fmt = st.selectbox(f"Format for {title}", formats, key=title)
                data = save_image_to_format(im, fmt)
                st.download_button(f"Download {title}", data, f"{title.lower()}.{fmt.lower()}", f"image/{fmt.lower()}")
        else:
            out = process_image(img, mode, severity)
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Original")
                st.image(img, width="stretch")
            with c2:
                st.subheader(f"{mode} ({int(severity*100)}%)")
                st.image(out, width="stretch")

            fmt = st.selectbox("Download format", formats)
            data = save_image_to_format(out, fmt)
            st.download_button(f"Download as {fmt}", data, f"{mode.lower()}.{fmt.lower()}", f"image/{fmt.lower()}")

    elif input_method == "Upload Video":
        uploaded_video = st.file_uploader("Upload video file", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_video is not None:
            # Generate a unique key for the uploaded file
            file_key = f"video_{uploaded_video.name}_{uploaded_video.size}"
            
            # Save uploaded video temporarily if not already saved in session
            if "input_video_path" not in st.session_state or st.session_state.get("current_video_key") != file_key:
                temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_input.write(uploaded_video.read())
                temp_input.close()
                st.session_state.input_video_path = temp_input.name
                st.session_state.current_video_key = file_key
                # Clear previous outputs when new video is uploaded
                st.session_state.processed_videos = {}
            
            input_path = st.session_state.input_video_path

            if all_view:
                # Key for this specific processing mode
                process_key = "all_types"
                
                if process_key in st.session_state.get("processed_videos", {}):
                    # Show cached results
                    st.success("Videos processed successfully! (Loaded from cache)")
                    outputs = st.session_state.processed_videos[process_key]
                    for vtype, output_path in outputs.items():
                        st.subheader(f"{vtype} Vision")
                        if os.path.exists(output_path):
                            with open(output_path, "rb") as f:
                                video_bytes = f.read()
                            st.video(video_bytes)
                            st.download_button(
                                f"Download {vtype} Video",
                                video_bytes,
                                f"colorblind_simulation_{vtype.lower()}.mp4",
                                "video/mp4",
                                key=f"download_{vtype}"
                            )
                        else:
                            st.warning(f"Video file for {vtype} not found. Please re-process.")
                            
                    if st.button("Process Again"):
                        # Clear cache for this key to force re-processing
                        del st.session_state.processed_videos[process_key]
                        st.rerun()

                elif st.button("Process Video (All Types)"):
                    st.info("Processing video for all vision types... This may take a while depending on video length.")
                    
                    # Create output paths
                    temp_outputs = {}
                    vision_types_for_video = ["Normal", "Protanopia", "Deuteranopia", "Tritanopia", "Achromatopsia"]
                    for vtype in vision_types_for_video:
                        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        temp_output.close()
                        temp_outputs[vtype] = temp_output.name
                    
                    try:
                        process_video_file_all_types(input_path, temp_outputs)
                        
                        # Initialize cache if needed
                        if "processed_videos" not in st.session_state:
                            st.session_state.processed_videos = {}
                        
                        # Store in session state
                        st.session_state.processed_videos[process_key] = temp_outputs
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing video: {e}")
                        # Cleanup on failure
                        for path in temp_outputs.values():
                            try:
                                os.unlink(path)
                            except:
                                pass

            else:
                # Single mode processing
                process_key = f"single_{mode}_{severity}"
                
                if process_key in st.session_state.get("processed_videos", {}):
                    # Show cached result
                    st.success("Video processed successfully! (Loaded from cache)")
                    output_path = st.session_state.processed_videos[process_key]
                    
                    if os.path.exists(output_path):
                        with open(output_path, "rb") as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                        st.download_button(
                            "Download Processed Video",
                            video_bytes,
                            f"colorblind_simulation_{mode.lower()}.mp4",
                            "video/mp4"
                        )
                    else:
                        st.warning("Video file not found. Please re-process.")

                    if st.button("Process Again"):
                        del st.session_state.processed_videos[process_key]
                        st.rerun()
                
                elif st.button("Process Video"):
                    st.info("Processing video... This may take a while depending on video length.")
                    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    temp_output.close()
                    
                    try:
                        process_video_file(input_path, temp_output.name, mode, severity)
                        
                        if "processed_videos" not in st.session_state:
                            st.session_state.processed_videos = {}
                        
                        st.session_state.processed_videos[process_key] = temp_output.name
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing video: {e}")
                        try:
                            os.unlink(temp_output.name)
                        except:
                            pass

    elif input_method == "Live Webcam":
        st.subheader("Live Webcam Simulation")
        st.info("Click 'Start Webcam' to begin live color blindness simulation.")
        
        if all_view:
            # Create placeholders for all vision types
            st.write("**All Vision Types**")
            vision_types_for_webcam = ["Normal Vision", "Protanopia", "Deuteranopia", "Tritanopia", "Achromatopsia"]
            
            # Create grid layout
            cols = st.columns(3)
            placeholders = {}
            for i, vtype in enumerate(vision_types_for_webcam):
                with cols[i % 3]:
                    st.write(f"**{vtype}**")
                    placeholders[vtype] = st.empty()
        else:
            # Create placeholders for webcam display
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original**")
                original_placeholder = st.empty()
                
            with col2:
                st.write(f"**{mode} ({int(severity*100)}%)**")
                processed_placeholder = st.empty()
        
        # Webcam control buttons
        start_webcam = st.button("Start Webcam")
        stop_webcam = st.button("Stop Webcam")
        
        # Initialize session state for webcam
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False
        
        if start_webcam:
            st.session_state.webcam_active = True
            
        if stop_webcam:
            st.session_state.webcam_active = False
        
        # Webcam processing
        if st.session_state.webcam_active:
            try:
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("Could not access webcam. Please ensure your webcam is connected and not being used by another application.")
                    st.session_state.webcam_active = False
                else:
                    frame_count = 0
                    while st.session_state.webcam_active and frame_count < 300:  # Limit to prevent infinite loop
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to capture frame from webcam")
                            break
                        
                        if all_view:
                            # Process frame for all vision types
                            original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            # Display all types
                            for vtype in vision_types_for_webcam:
                                if vtype == "Normal Vision":
                                    display_frame = original_rgb
                                else:
                                    processed_frame = process_frame_cv2(frame, vtype, 1.0)
                                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                                
                                placeholders[vtype].image(display_frame, channels="RGB", width="stretch")
                        else:
                            # Process frame
                            processed_frame = process_frame_cv2(frame, mode, severity)
                            
                            # Convert frames for display
                            original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            
                            # Display frames
                            original_placeholder.image(original_rgb, channels="RGB", width="stretch")
                            processed_placeholder.image(processed_rgb, channels="RGB", width="stretch")
                        
                        frame_count += 1
                        time.sleep(0.033)  # ~30 FPS
                    
                    cap.release()
                    
            except Exception as e:
                st.error(f"Webcam error: {e}")
                st.session_state.webcam_active = False


if __name__ == "__main__":
    main()
