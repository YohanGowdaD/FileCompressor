import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET

import streamlit as st
from PIL import Image
from pikepdf import Pdf, ObjectStreamMode, Name
import fitz  # PyMuPDF for advanced PDF operations

# ---------------------------
# Helpers: formatting & sizes
# ---------------------------
def human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"

# ---------------------------------
# Enhanced PDF compression
# ---------------------------------
def compress_pdf_lossless(pdf_bytes: bytes, linearize: bool = True) -> bytes:
    """Enhanced lossless PDF optimization using pikepdf."""
    inp = io.BytesIO(pdf_bytes)
    out = io.BytesIO()
    with Pdf.open(inp) as pdf:
        # Remove unnecessary metadata
        if pdf.trailer.get("/Info"):
            info = pdf.trailer["/Info"]
            # Keep only essential metadata
            essential = ["/Title", "/Author"]
            keys_to_remove = [k for k in info.keys() if k not in essential]
            for k in keys_to_remove:
                del info[k]
        
        pdf.save(
            out,
            compress_streams=True,
            object_stream_mode=ObjectStreamMode.generate,
            preserve_pdfa=True,
            linearize=linearize,
            stream_decode_level=None  # Let pikepdf decide best compression
        )
    return out.getvalue()

def compress_pdf_aggressive(pdf_bytes: bytes, target_mb: float = None, preserve_quality: bool = True) -> bytes:
    """
    Aggressive PDF compression with font subsetting, image optimization, and metadata removal.
    Uses PyMuPDF for advanced operations.
    """
    doc = fitz.open("pdf", pdf_bytes)
    
    # Create output buffer
    out = io.BytesIO()
    
    # Compression settings based on quality preservation
    if preserve_quality:
        # High quality: minimal image degradation
        garbage_opts = 3  # Remove unused objects
        deflate_opts = True
        image_quality = 95
    else:
        # More aggressive
        garbage_opts = 4  # More aggressive cleanup
        deflate_opts = True
        image_quality = 85
    
    # Process each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Get all images on the page
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            
            try:
                # Extract image
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Open with PIL for recompression
                img = Image.open(io.BytesIO(image_bytes))
                
                # Calculate optimal dimensions based on target
                width, height = img.size
                max_dimension = 2400 if preserve_quality else 1800
                
                if width > max_dimension or height > max_dimension:
                    # Resize maintaining aspect ratio
                    if width > height:
                        new_width = max_dimension
                        new_height = int(height * (max_dimension / width))
                    else:
                        new_height = max_dimension
                        new_width = int(width * (max_dimension / height))
                    
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Compress image
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=image_quality, optimize=True)
                compressed_bytes = img_buffer.getvalue()
                
                # Replace image if compression achieved size reduction
                if len(compressed_bytes) < len(image_bytes):
                    page.replace_image(xref, stream=compressed_bytes)
                    
            except Exception as e:
                # Skip problematic images
                continue
    
    # Save with optimization
    doc.save(
        out,
        garbage=garbage_opts,
        deflate=deflate_opts,
        clean=True,  # Clean up content streams
        linear=True,  # Linearize for fast web view
    )
    doc.close()
    
    # Second pass with pikepdf for additional compression
    out.seek(0)
    return compress_pdf_lossless(out.read(), linearize=True)

def which_gs():
    """Find Ghostscript executable."""
    candidates = ["gs", "gswin64c", "gswin32c"]
    for c in candidates:
        if shutil.which(c):
            return c
    return None

def compress_pdf_with_gs(pdf_bytes: bytes, preset: str = "/ebook") -> bytes:
    """Use Ghostscript for additional compression."""
    gs_exec = which_gs()
    if not gs_exec:
        return None

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.pdf")
        out_path = os.path.join(td, "out.pdf")
        with open(in_path, "wb") as f:
            f.write(pdf_bytes)

        cmd = [
            gs_exec,
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            f"-dPDFSETTINGS={preset}",
            "-dNOPAUSE",
            "-dQUIET",
            "-dBATCH",
            "-dDetectDuplicateImages=true",
            "-dCompressFonts=true",
            "-dSubsetFonts=true",
            "-dEmbedAllFonts=false",
            f"-sOutputFile={out_path}",
            in_path,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(out_path, "rb") as f:
                return f.read()
        except Exception:
            return None

def compress_pdf_to_target(pdf_bytes: bytes, target_mb: float, preserve_quality: bool = True) -> bytes:
    """Intelligently compress PDF to reach target size."""
    
    # Start with aggressive compression
    compressed = compress_pdf_aggressive(pdf_bytes, target_mb, preserve_quality)
    
    if len(compressed) <= target_mb * 1024 * 1024:
        return compressed
    
    # Try Ghostscript if available
    gs_exec = which_gs()
    if gs_exec:
        presets = ["/printer", "/ebook", "/screen"]
        best = compressed
        
        for preset in presets:
            gs_result = compress_pdf_with_gs(compressed, preset)
            if gs_result and len(gs_result) < len(best):
                best = gs_result
                if len(best) <= target_mb * 1024 * 1024:
                    return best
        
        return best
    
    return compressed

# ---------------------------------------------------------
# Enhanced DOCX compression
# ---------------------------------------------------------
def remove_docx_metadata(extract_dir: str) -> None:
    """Remove unnecessary metadata from DOCX to reduce size."""
    try:
        # Remove core properties
        core_props = os.path.join(extract_dir, "docProps", "core.xml")
        if os.path.exists(core_props):
            tree = ET.parse(core_props)
            root = tree.getroot()
            # Keep only essential properties
            for child in list(root):
                if not any(x in child.tag for x in ['title', 'creator']):
                    root.remove(child)
            tree.write(core_props)
        
        # Remove app properties (except essentials)
        app_props = os.path.join(extract_dir, "docProps", "app.xml")
        if os.path.exists(app_props):
            tree = ET.parse(app_props)
            root = tree.getroot()
            for child in list(root):
                if any(x in child.tag for x in ['TotalTime', 'Company', 'Manager', 'HyperlinksChanged']):
                    root.remove(child)
            tree.write(app_props)
            
    except Exception:
        pass

def optimize_image_smart(path: str, jpeg_quality: int = 90, preserve_quality: bool = True) -> None:
    """
    Smart image optimization that preserves visual quality while reducing size.
    """
    try:
        img = Image.open(path)
        original_size = os.path.getsize(path)
        
        # Get image info
        width, height = img.size
        img_format = img.format
        
        # Determine optimal max dimension based on quality preference
        if preserve_quality:
            max_dimension = 3000  # High quality
            jpeg_quality = max(jpeg_quality, 92)
        else:
            max_dimension = 2000  # Balanced
        
        # Resize if too large
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert mode intelligently
        if img.mode == 'RGBA':
            # Check if alpha channel is actually used
            if img.getextrema()[3] == (255, 255):
                img = img.convert('RGB')
            else:
                # Keep as PNG for transparency
                img.save(path, format='PNG', optimize=True, compress_level=9)
                return
        elif img.mode == 'P':
            img = img.convert('RGB')
        elif img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        # Save optimized
        if img_format in ['JPEG', 'JPG'] or img.mode == 'RGB':
            img.save(path, format='JPEG', quality=jpeg_quality, optimize=True, progressive=True)
        elif img_format == 'PNG':
            img.save(path, format='PNG', optimize=True, compress_level=9)
        else:
            img.save(path, format='JPEG', quality=jpeg_quality, optimize=True)
            
    except Exception:
        pass

def rezip_docx_lossless(docx_bytes: bytes, zip_level: int = 9) -> bytes:
    """Lossless DOCX compression through re-zipping."""
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.docx")
        out_path = os.path.join(td, "out.docx")

        with open(in_path, "wb") as f:
            f.write(docx_bytes)

        extract_dir = os.path.join(td, "unzipped")
        os.makedirs(extract_dir, exist_ok=True)
        
        with ZipFile(in_path, "r") as zin:
            zin.extractall(extract_dir)
        
        # Remove metadata for size reduction
        remove_docx_metadata(extract_dir)

        # Repack with maximum compression
        with ZipFile(out_path, "w", compression=ZIP_DEFLATED, compresslevel=zip_level) as zout:
            for root, _, files in os.walk(extract_dir):
                for name in files:
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, extract_dir)
                    zout.write(full, rel)

        with open(out_path, "rb") as f:
            return f.read()

def compress_docx_aggressive(docx_bytes: bytes, jpeg_quality: int, preserve_quality: bool = True) -> bytes:
    """
    Aggressive DOCX compression with smart image optimization and metadata removal.
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
        
        # Remove metadata
        remove_docx_metadata(extract_dir)

        # Optimize images
        media_dir = os.path.join(extract_dir, "word", "media")
        if os.path.isdir(media_dir):
            for name in os.listdir(media_dir):
                p = os.path.join(media_dir, name)
                if os.path.isfile(p):
                    optimize_image_smart(p, jpeg_quality=jpeg_quality, preserve_quality=preserve_quality)

        # Repack with maximum compression
        with ZipFile(out_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as zout:
            for root, _, files in os.walk(extract_dir):
                for name in files:
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, extract_dir)
                    zout.write(full, rel)

        with open(out_path, "rb") as f:
            return f.read()

def compress_docx_to_target(docx_bytes: bytes, target_mb: float, preserve_quality: bool = True) -> bytes:
    """
    Intelligently compress DOCX to reach target size while preserving quality.
    """
    # Start with lossless
    compressed = rezip_docx_lossless(docx_bytes, zip_level=9)
    
    if len(compressed) <= target_mb * 1024 * 1024:
        return compressed
    
    # Try different quality levels
    if preserve_quality:
        quality_steps = [95, 92, 88, 85]
    else:
        quality_steps = [90, 85, 80, 75, 70, 65]
    
    best = compressed
    for quality in quality_steps:
        result = compress_docx_aggressive(docx_bytes, jpeg_quality=quality, preserve_quality=preserve_quality)
        if len(result) < len(best):
            best = result
        if len(result) <= target_mb * 1024 * 1024:
            return result
    
    return best

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Advanced Document Compressor", page_icon="🗜️", layout="centered")

st.title("🗜️ Advanced PDF & Word Compressor")
st.markdown("""
**Enhanced compression with intelligent quality preservation**
- 📦 Removes unnecessary metadata & fonts
- 🖼️ Smart image optimization
- 🎯 Target size with minimal quality loss
""")

uploaded = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])

# Compression mode selection
mode = st.radio(
    "Compression mode",
    ["Smart Lossless (Best Quality)", "Target Size (Quality Preserved)", "Maximum Compression"],
    index=0,
    help="Smart Lossless: Removes bloat without affecting quality\nTarget Size: Reaches your size goal with minimal quality impact\nMaximum: Smallest possible file"
)

col1, col2 = st.columns(2)
with col1:
    target_mb = st.number_input(
        "Target size (MB)", 
        min_value=0.1, 
        max_value=500.0, 
        value=2.0, 
        step=0.1,
        disabled=(mode == "Smart Lossless (Best Quality)")
    )

with col2:
    preserve_quality = st.checkbox(
        "Preserve Visual Quality",
        value=(mode != "Maximum Compression"),
        help="When enabled, uses higher quality settings for images"
    )

st.caption("💡 Tip: 'Target Size' mode intelligently compresses to your desired size while keeping documents visually sharp.")

if uploaded is not None:
    data = uploaded.read()
    in_size = len(data)

    st.write(f"**Original:** {uploaded.name} · {human_size(in_size)}")

    file_suffix = Path(uploaded.name).suffix.lower()

    if st.button("🚀 Compress", type="primary", use_container_width=True):
        with st.spinner("Compressing... This may take a moment for large files."):
            try:
                if file_suffix == ".pdf":
                    if mode == "Smart Lossless (Best Quality)":
                        out_bytes = compress_pdf_lossless(data, linearize=True)
                    elif mode == "Target Size (Quality Preserved)":
                        out_bytes = compress_pdf_to_target(data, target_mb=target_mb, preserve_quality=True)
                    else:  # Maximum Compression
                        out_bytes = compress_pdf_to_target(data, target_mb=target_mb, preserve_quality=False)

                    out_name = Path(uploaded.name).stem + ".compressed.pdf"

                elif file_suffix == ".docx":
                    if mode == "Smart Lossless (Best Quality)":
                        out_bytes = rezip_docx_lossless(data, zip_level=9)
                    elif mode == "Target Size (Quality Preserved)":
                        out_bytes = compress_docx_to_target(data, target_mb=target_mb, preserve_quality=True)
                    else:  # Maximum Compression
                        out_bytes = compress_docx_to_target(data, target_mb=target_mb, preserve_quality=False)

                    out_name = Path(uploaded.name).stem + ".compressed.docx"
                else:
                    st.error("Unsupported file type. Please upload a .pdf or .docx.")
                    st.stop()

                out_size = len(out_bytes)
                ratio = (1 - out_size / in_size) * 100 if in_size else 0.0

                # Display results
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Original", human_size(in_size))
                with col_b:
                    st.metric("Compressed", human_size(out_size))
                with col_c:
                    st.metric("Saved", f"{ratio:.1f}%")

                st.success("✅ Compression completed successfully!")
                
                st.download_button(
                    "⬇️ Download Compressed File",
                    data=out_bytes,
                    file_name=out_name,
                    mime="application/octet-stream",
                    use_container_width=True
                )

                # # Details
                # with st.expander("📊 Technical Details"):
                #     st.json({
                #         "original_size_bytes": in_size,
                #         "compressed_size_bytes": out_size,
                #         "compression_ratio": f"{ratio:.2f}%",
                #         "ghostscript_available": which_gs() is not None,
                #         "mode_used": mode,
                #         "quality_preserved": preserve_quality
                #     })

                if file_suffix == ".pdf" and which_gs() is None:
                    st.info("💡 Install **Ghostscript** for even better PDF compression in aggressive modes.")

            except Exception as e:
                st.error(f"❌ Compression failed: {e}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>Built with ❤️ for efficient document compression | Supports PDF & DOCX | Yohan Gowda D</small>
</div>
""", unsafe_allow_html=True)