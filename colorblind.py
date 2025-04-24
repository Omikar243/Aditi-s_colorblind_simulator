import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import tempfile
import os
import time
import matplotlib.pyplot as plt

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


def apply_protanopia(image: Image.Image) -> Image.Image:
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
    arr = np.array(image, dtype=np.float32)
    flat = arr.reshape(-1, 3)
    sim = flat.dot(M.T)
    sim = np.clip(sim, 0, 255).astype(np.uint8)
    return Image.fromarray(sim.reshape(arr.shape))


def apply_deuteranopia(image: Image.Image) -> Image.Image:
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
    arr = np.array(image, dtype=np.float32)
    flat = arr.reshape(-1, 3)
    sim = flat.dot(M.T)
    sim = np.clip(sim, 0, 255).astype(np.uint8)
    return Image.fromarray(sim.reshape(arr.shape))


def apply_tritanopia(image: Image.Image) -> Image.Image:
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
    arr = np.array(image, dtype=np.float32)
    flat = arr.reshape(-1, 3)
    sim = flat.dot(M.T)
    sim = np.clip(sim, 0, 255).astype(np.uint8)
    return Image.fromarray(sim.reshape(arr.shape))


def apply_achromatopsia(image: Image.Image) -> Image.Image:
    """Simulate complete color blindness by converting to grayscale."""
    return image.convert("L").convert("RGB")


def process_image(image: Image.Image, mode: str) -> Image.Image:
    if mode == "Protanopia":
        return apply_protanopia(image)
    elif mode == "Deuteranopia":
        return apply_deuteranopia(image)
    elif mode == "Tritanopia":
        return apply_tritanopia(image)
    elif mode == "Achromatopsia":
        return apply_achromatopsia(image)
    else:
        return image


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


def main():
    st.set_page_config(page_title="Colorblind Simulator", layout="wide")
    st.title("Aditi's Colorblind Simulator")
    st.write("""
    Upload an image and see how it appears under different color vision deficiencies.
    You can also download the result in several formats.
    """)

    if not HEIC_SUPPORT:
        st.warning("Install `pillow-heif` for HEIC support: `pip install pillow-heif`")

    uploaded = st.file_uploader("Upload image (JPEG, PNG, HEIC…)", type=["jpg","jpeg","png","heic"])
    mode = st.selectbox("Select vision type", ["Normal Vision","Protanopia","Deuteranopia","Tritanopia","Achromatopsia"])
    all_view = st.checkbox("Show all types side by side")

    formats = ["PNG","JPEG","WEBP","BMP","GIF"]
    if HEIC_SUPPORT: formats.append("HEIC")

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
            "Protanopia": apply_protanopia(img),
            "Deuteranopia": apply_deuteranopia(img),
            "Tritanopia": apply_tritanopia(img),
            "Achromatopsia": apply_achromatopsia(img)
        }
        cols = st.columns(3)
        for i,(title,im) in enumerate(imgs.items()):
            with cols[i%3]:
                st.write(title)
                st.image(im, use_container_width=True)
        st.markdown("---")
        for title,im in imgs.items():
            fmt = st.selectbox(f"Format for {title}", formats, key=title)
            data = save_image_to_format(im, fmt)
            st.download_button(f"Download {title}", data, f"{title.lower()}.{fmt.lower()}", f"image/{fmt.lower()}")
    else:
        out = process_image(img, mode)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Original")
            st.image(img, use_container_width=True)
        with c2:
            st.subheader(mode)
            st.image(out, use_container_width=True)

        fmt = st.selectbox("Download format", formats)
        data = save_image_to_format(out, fmt)
        st.download_button(f"Download as {fmt}", data, f"{mode.lower()}.{fmt.lower()}", f"image/{fmt.lower()}")

if __name__ == "__main__":
    main()
