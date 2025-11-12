import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

import streamlit as st
from PIL import Image
from pikepdf import Pdf, ObjectStreamMode

# ---------------------------
# Helpers: formatting & sizes
# ---------------------------
def human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"

def get_bytes(io_or_bytes) -> bytes:
    if isinstance(io_or_bytes, (bytes, bytearray)):
        return bytes(io_or_bytes)
    if hasattr(io_or_bytes, "getvalue"):
        return io_or_bytes.getvalue()
    raise TypeError("Unsupported type for get_bytes")

# ---------------------------------
# Lossless PDF compression (pikepdf)
# ---------------------------------
def compress_pdf_lossless(pdf_bytes: bytes, linearize: bool = True) -> bytes:
    """Lossless PDF size optimization using pikepdf/QPDF."""
    inp = io.BytesIO(pdf_bytes)
    out = io.BytesIO()
    with Pdf.open(inp) as pdf:
        pdf.save(
            out,
            compress_streams=True,               # deflate streams if beneficial
            object_stream_mode=ObjectStreamMode.generate,  # pack small objects
            preserve_pdfa=True,                  # try to preserve PDF/A if present
            linearize=linearize                  # fast web view
        )
    return out.getvalue()

# ---------------------------------------------------
# Aggressive PDF compression via Ghostscript (if any)
# ---------------------------------------------------
def which_gs():
    # Different executable names depending on platform
    candidates = ["gs", "gswin64c", "gswin32c"]
    for c in candidates:
        if shutil.which(c):
            return c
    return None

def compress_pdf_with_gs(pdf_bytes: bytes, preset: str = "/ebook") -> bytes:
    """
    Use Ghostscript presets:
      /screen  -> heavy compression, lowest quality
      /ebook   -> medium compression, good for screen reading
      /printer -> higher quality, bigger file
      /prepress-> highest quality (downsamples less)
    """
    gs_exec = which_gs()
    if not gs_exec:
        return None  # signal that GS is unavailable

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.pdf")
        out_path = os.path.join(td, "out.pdf")
        with open(in_path, "wb") as f:
            f.write(pdf_bytes)

        # Note: -dCompatibilityLevel=1.4 ensures broader compatibility & size savings
        cmd = [
            gs_exec,
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            f"-dPDFSETTINGS={preset}",
            "-dNOPAUSE",
            "-dQUIET",
            "-dBATCH",
            f"-sOutputFile={out_path}",
            in_path,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(out_path, "rb") as f:
                return f.read()
        except Exception:
            return None

def compress_pdf_to_target(pdf_bytes: bytes, target_mb: float) -> bytes:
    """
    Try progressively stronger Ghostscript presets to get under target size.
    Falls back to lossless if GS isn't available or can't meet target.
    """
    base_lossless = compress_pdf_lossless(pdf_bytes)
    if len(base_lossless) <= target_mb * 1024 * 1024:
        return base_lossless

    gs_exec = which_gs()
    if not gs_exec:
        # Can't do aggressive; return best lossless
        return base_lossless

    # Try presets from higher quality to more aggressive
    presets = ["/prepress", "/printer", "/ebook", "/screen"]
    best = base_lossless
    for p in presets:
        out = compress_pdf_with_gs(pdf_bytes, p)
        if out is None:
            continue
        if len(out) < len(best):
            best = out
        if len(out) <= target_mb * 1024 * 1024:
            return out
    return best

# ---------------------------------------------------------
# DOCX compression: re-zip losslessly + (optional) images
# ---------------------------------------------------------
def rezip_docx_lossless(docx_bytes: bytes, zip_level: int = 9) -> bytes:
    """
    DOCX is a ZIP. Re-zip with maximum DEFLATE compression for lossless savings.
    """
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.docx")
        out_path = os.path.join(td, "out.docx")

        with open(in_path, "wb") as f:
            f.write(docx_bytes)

        extract_dir = os.path.join(td, "unzipped")
        os.makedirs(extract_dir, exist_ok=True)
        with ZipFile(in_path, "r") as zin:
            zin.extractall(extract_dir)

        # Repack with max compression
        with ZipFile(out_path, "w", compression=ZIP_DEFLATED, compresslevel=zip_level) as zout:
            for root, _, files in os.walk(extract_dir):
                for name in files:
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, extract_dir)
                    zout.write(full, rel)

        with open(out_path, "rb") as f:
            return f.read()

def optimize_image(path: str, jpeg_quality: int = 90) -> None:
    """
    Recompress images inside DOCX media folder. This MAY reduce quality for JPEGs.
    PNGs are saved with optimize flag (lossless-ish).
    """
    try:
        img = Image.open(path)
        img_format = img.format
        # Convert mode to something encodable if needed
        if img.mode in ("P", "RGBA"):
            # Prefer PNG when alpha present; keep PNG for non-photographic images
            if img_format != "PNG":
                img = img.convert("RGBA")
                img_format = "PNG"
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        if img_format == "JPEG":
            img.save(path, format="JPEG", quality=jpeg_quality, optimize=True, progressive=True)
        elif img_format == "PNG":
            # PNG optimize; consider quantization for further size (but that may alter colors)
            img.save(path, format="PNG", optimize=True)
        else:
            # For other formats, try saving as original (may not reduce size much)
            img.save(path, format=img_format)
    except Exception:
        # If any error, leave the original file as-is
        pass

def compress_docx_aggressive(docx_bytes: bytes, zip_level: int, jpeg_quality: int) -> bytes:
    """
    Re-zip and recompress embedded images. May reduce visual quality (JPEG).
    """
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.docx")
        out_path = os.path.join(td, "out.docx")

        with open(in_path, "wb") as f:
            f.write(docx_bytes)

        extract_dir = os.path.join(td, "unzipped")
        os.makedirs(extract_dir, exist_ok=True)
        with ZipFile(in_path, "r") as zin:
            zin.extractall(extract_dir)

        # Recompress images in word/media/*
        media_dir = os.path.join(extract_dir, "word", "media")
        if os.path.isdir(media_dir):
            for name in os.listdir(media_dir):
                p = os.path.join(media_dir, name)
                if os.path.isfile(p):
                    optimize_image(p, jpeg_quality=jpeg_quality)

        # Repack
        with ZipFile(out_path, "w", compression=ZIP_DEFLATED, compresslevel=zip_level) as zout:
            for root, _, files in os.walk(extract_dir):
                for name in files:
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, extract_dir)
                    zout.write(full, rel)

        with open(out_path, "rb") as f:
            return f.read()

def compress_docx_to_target(docx_bytes: bytes, target_mb: float, start_quality: int = 95) -> bytes:
    """
    Attempt to reach a target size by gradually lowering JPEG quality of images.
    Lossless rezip first; then step down JPEG quality.
    """
    best = rezip_docx_lossless(docx_bytes, zip_level=9)
    if len(best) <= target_mb * 1024 * 1024:
        return best

    quality_steps = [95, 90, 85, 80, 75, 70, 60, 50]
    for q in quality_steps:
        out = compress_docx_aggressive(best, zip_level=9, jpeg_quality=q)
        if len(out) < len(best):
            best = out
        if len(out) <= target_mb * 1024 * 1024:
            return out
    return best

# -------------------------
# Streamlit UI / App logic
# -------------------------
st.set_page_config(page_title="PDF & DOCX Compressor", page_icon="🗜️", layout="centered")

st.title("🗜️ PDF & Word (.docx) Compressor")
st.write("**Default is lossless compression** (no visual quality change). Choose *Aggressive / Target size* if you need much smaller files (may reduce image quality).")

uploaded = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])

mode = st.radio(
    "Compression mode",
    ["Lossless (keep same quality)", "Aggressive / Target size"],
    index=0,
    help="Lossless: structure/image streams recompressed without changing quality.\nAggressive: tries to hit a target size using image downsampling/recompression."
)

col1, col2 = st.columns(2)
with col1:
    target_mb = st.number_input("Target size (MB) — used in Aggressive mode", min_value=0.1, max_value=500.0, value=2.0, step=0.1)
with col2:
    docx_jpeg_quality = st.slider("DOCX JPEG quality (Aggressive)", min_value=50, max_value=100, value=90, step=1)

st.caption("Tip: Lower target size or JPEG quality yields smaller files but can reduce visual fidelity (Aggressive mode).")

if uploaded is not None:
    data = uploaded.read()
    in_size = len(data)

    st.write(f"**Original:** {uploaded.name} · {human_size(in_size)}")

    file_suffix = Path(uploaded.name).suffix.lower()

    if st.button("Compress"):
        try:
            if file_suffix == ".pdf":
                if mode.startswith("Lossless"):
                    out_bytes = compress_pdf_lossless(data, linearize=True)
                else:
                    # target size path; try GS presets, fall back to lossless if not available
                    out_bytes = compress_pdf_to_target(data, target_mb=target_mb)
                    # If GS not available and lossless didn't help, you still get the best achievable

                out_name = Path(uploaded.name).with_suffix(".compressed.pdf").name

            elif file_suffix == ".docx":
                if mode.startswith("Lossless"):
                    out_bytes = rezip_docx_lossless(data, zip_level=9)
                else:
                    # Try to meet target by stepping JPEG quality
                    # Start from user-provided quality hint
                    temp = compress_docx_aggressive(data, zip_level=9, jpeg_quality=docx_jpeg_quality)
                    if len(temp) > target_mb * 1024 * 1024:
                        # iterate down if still above target
                        out_bytes = compress_docx_to_target(temp, target_mb=target_mb, start_quality=docx_jpeg_quality)
                    else:
                        out_bytes = temp

                out_name = Path(uploaded.name).with_suffix(".compressed.docx").name
            else:
                st.error("Unsupported file type. Please upload a .pdf or .docx.")
                st.stop()

            out_size = len(out_bytes)
            ratio = (1 - out_size / in_size) * 100 if in_size else 0.0

            st.success(f"Done! New size: **{human_size(out_size)}** (saved {ratio:.1f}%)")
            st.download_button(
                "⬇️ Download compressed file",
                data=out_bytes,
                file_name=out_name,
                mime="application/octet-stream"
            )

            # Extra diagnostics
            with st.expander("Show details"):
                st.write({
                    "original_size_bytes": in_size,
                    "compressed_size_bytes": out_size,
                    "saved_percent": round(ratio, 2),
                    "ghostscript_available": which_gs() is not None
                })

            if file_suffix == ".pdf" and mode.startswith("Aggressive") and which_gs() is None:
                st.info("For even smaller PDFs in Aggressive mode, install **Ghostscript** (`gs`). The app will auto-detect it if available.")

        except Exception as e:
            st.error(f"Compression failed: {e}")
            st.stop()

# Footer
st.markdown("---")
st.markdown(
    "Made for **fast, safe compression**. Lossless mode preserves visual quality. "
    "Aggressive mode targets smaller sizes and may reduce image quality."
)
