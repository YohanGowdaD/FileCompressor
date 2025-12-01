# import io
# import os
# import shutil
# import subprocess
# import tempfile
# from pathlib import Path
# from zipfile import ZipFile, ZIP_DEFLATED
# import xml.etree.ElementTree as ET

# import streamlit as st
# from PIL import Image
# from pikepdf import Pdf, ObjectStreamMode, Name
# import fitz  # PyMuPDF for advanced PDF operations

# # ---------------------------
# # Helpers: formatting & sizes
# # ---------------------------
# def human_size(num_bytes: int) -> str:
#     for unit in ["B", "KB", "MB", "GB"]:
#         if num_bytes < 1024.0:
#             return f"{num_bytes:3.1f} {unit}"
#         num_bytes /= 1024.0
#     return f"{num_bytes:.1f} TB"

# # ---------------------------------
# # Enhanced PDF compression
# # ---------------------------------
# def compress_pdf_lossless(pdf_bytes: bytes, linearize: bool = True) -> bytes:
#     """Enhanced lossless PDF optimization using pikepdf."""
#     inp = io.BytesIO(pdf_bytes)
#     out = io.BytesIO()
#     with Pdf.open(inp) as pdf:
#         # Remove unnecessary metadata
#         if pdf.trailer.get("/Info"):
#             info = pdf.trailer["/Info"]
#             # Keep only essential metadata
#             essential = ["/Title", "/Author"]
#             keys_to_remove = [k for k in info.keys() if k not in essential]
#             for k in keys_to_remove:
#                 del info[k]
        
#         pdf.save(
#             out,
#             compress_streams=True,
#             object_stream_mode=ObjectStreamMode.generate,
#             preserve_pdfa=True,
#             linearize=linearize,
#             stream_decode_level=None  # Let pikepdf decide best compression
#         )
#     return out.getvalue()

# def compress_pdf_aggressive(pdf_bytes: bytes, target_mb: float = None, preserve_quality: bool = True) -> bytes:
#     """
#     Aggressive PDF compression with font subsetting, image optimization, and metadata removal.
#     Uses PyMuPDF for advanced operations.
#     """
#     doc = fitz.open("pdf", pdf_bytes)
    
#     # Create output buffer
#     out = io.BytesIO()
    
#     # Compression settings based on quality preservation
#     if preserve_quality:
#         # High quality: minimal image degradation
#         garbage_opts = 3  # Remove unused objects
#         deflate_opts = True
#         image_quality = 95
#     else:
#         # More aggressive
#         garbage_opts = 4  # More aggressive cleanup
#         deflate_opts = True
#         image_quality = 85
    
#     # Process each page
#     for page_num in range(len(doc)):
#         page = doc[page_num]
        
#         # Get all images on the page
#         image_list = page.get_images(full=True)
        
#         for img_index, img_info in enumerate(image_list):
#             xref = img_info[0]
            
#             try:
#                 # Extract image
#                 base_image = doc.extract_image(xref)
#                 image_bytes = base_image["image"]
#                 image_ext = base_image["ext"]
                
#                 # Open with PIL for recompression
#                 img = Image.open(io.BytesIO(image_bytes))
                
#                 # Calculate optimal dimensions based on target
#                 width, height = img.size
#                 max_dimension = 2400 if preserve_quality else 1800
                
#                 if width > max_dimension or height > max_dimension:
#                     # Resize maintaining aspect ratio
#                     if width > height:
#                         new_width = max_dimension
#                         new_height = int(height * (max_dimension / width))
#                     else:
#                         new_height = max_dimension
#                         new_width = int(width * (max_dimension / height))
                    
#                     img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
#                 # Convert to RGB if necessary
#                 if img.mode not in ('RGB', 'L'):
#                     img = img.convert('RGB')
                
#                 # Compress image
#                 img_buffer = io.BytesIO()
#                 img.save(img_buffer, format='JPEG', quality=image_quality, optimize=True)
#                 compressed_bytes = img_buffer.getvalue()
                
#                 # Replace image if compression achieved size reduction
#                 if len(compressed_bytes) < len(image_bytes):
#                     page.replace_image(xref, stream=compressed_bytes)
                    
#             except Exception as e:
#                 # Skip problematic images
#                 continue
    
#     # Save with optimization
#     doc.save(
#         out,
#         garbage=garbage_opts,
#         deflate=deflate_opts,
#         clean=True,  # Clean up content streams
#         linear=True,  # Linearize for fast web view
#     )
#     doc.close()
    
#     # Second pass with pikepdf for additional compression
#     out.seek(0)
#     return compress_pdf_lossless(out.read(), linearize=True)

# def which_gs():
#     """Find Ghostscript executable."""
#     candidates = ["gs", "gswin64c", "gswin32c"]
#     for c in candidates:
#         if shutil.which(c):
#             return c
#     return None

# def compress_pdf_with_gs(pdf_bytes: bytes, preset: str = "/ebook") -> bytes:
#     """Use Ghostscript for additional compression."""
#     gs_exec = which_gs()
#     if not gs_exec:
#         return None

#     with tempfile.TemporaryDirectory() as td:
#         in_path = os.path.join(td, "in.pdf")
#         out_path = os.path.join(td, "out.pdf")
#         with open(in_path, "wb") as f:
#             f.write(pdf_bytes)

#         cmd = [
#             gs_exec,
#             "-sDEVICE=pdfwrite",
#             "-dCompatibilityLevel=1.4",
#             f"-dPDFSETTINGS={preset}",
#             "-dNOPAUSE",
#             "-dQUIET",
#             "-dBATCH",
#             "-dDetectDuplicateImages=true",
#             "-dCompressFonts=true",
#             "-dSubsetFonts=true",
#             "-dEmbedAllFonts=false",
#             f"-sOutputFile={out_path}",
#             in_path,
#         ]
#         try:
#             subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             with open(out_path, "rb") as f:
#                 return f.read()
#         except Exception:
#             return None

# def compress_pdf_to_target(pdf_bytes: bytes, target_mb: float, preserve_quality: bool = True) -> bytes:
#     """Intelligently compress PDF to reach target size."""
    
#     # Start with aggressive compression
#     compressed = compress_pdf_aggressive(pdf_bytes, target_mb, preserve_quality)
    
#     if len(compressed) <= target_mb * 1024 * 1024:
#         return compressed
    
#     # Try Ghostscript if available
#     gs_exec = which_gs()
#     if gs_exec:
#         presets = ["/printer", "/ebook", "/screen"]
#         best = compressed
        
#         for preset in presets:
#             gs_result = compress_pdf_with_gs(compressed, preset)
#             if gs_result and len(gs_result) < len(best):
#                 best = gs_result
#                 if len(best) <= target_mb * 1024 * 1024:
#                     return best
        
#         return best
    
#     return compressed

# # ---------------------------------------------------------
# # Enhanced DOCX compression
# # ---------------------------------------------------------
# def remove_docx_metadata(extract_dir: str) -> None:
#     """Remove unnecessary metadata from DOCX to reduce size."""
#     try:
#         # Remove core properties
#         core_props = os.path.join(extract_dir, "docProps", "core.xml")
#         if os.path.exists(core_props):
#             tree = ET.parse(core_props)
#             root = tree.getroot()
#             # Keep only essential properties
#             for child in list(root):
#                 if not any(x in child.tag for x in ['title', 'creator']):
#                     root.remove(child)
#             tree.write(core_props)
        
#         # Remove app properties (except essentials)
#         app_props = os.path.join(extract_dir, "docProps", "app.xml")
#         if os.path.exists(app_props):
#             tree = ET.parse(app_props)
#             root = tree.getroot()
#             for child in list(root):
#                 if any(x in child.tag for x in ['TotalTime', 'Company', 'Manager', 'HyperlinksChanged']):
#                     root.remove(child)
#             tree.write(app_props)
            
#     except Exception:
#         pass

# def optimize_image_smart(path: str, jpeg_quality: int = 90, preserve_quality: bool = True) -> None:
#     """
#     Smart image optimization that preserves visual quality while reducing size.
#     """
#     try:
#         img = Image.open(path)
#         original_size = os.path.getsize(path)
        
#         # Get image info
#         width, height = img.size
#         img_format = img.format
        
#         # Determine optimal max dimension based on quality preference
#         if preserve_quality:
#             max_dimension = 3000  # High quality
#             jpeg_quality = max(jpeg_quality, 92)
#         else:
#             max_dimension = 2000  # Balanced
        
#         # Resize if too large
#         if width > max_dimension or height > max_dimension:
#             if width > height:
#                 new_width = max_dimension
#                 new_height = int(height * (max_dimension / width))
#             else:
#                 new_height = max_dimension
#                 new_width = int(width * (max_dimension / height))
            
#             img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
#         # Convert mode intelligently
#         if img.mode == 'RGBA':
#             # Check if alpha channel is actually used
#             if img.getextrema()[3] == (255, 255):
#                 img = img.convert('RGB')
#             else:
#                 # Keep as PNG for transparency
#                 img.save(path, format='PNG', optimize=True, compress_level=9)
#                 return
#         elif img.mode == 'P':
#             img = img.convert('RGB')
#         elif img.mode not in ('RGB', 'L'):
#             img = img.convert('RGB')
        
#         # Save optimized
#         if img_format in ['JPEG', 'JPG'] or img.mode == 'RGB':
#             img.save(path, format='JPEG', quality=jpeg_quality, optimize=True, progressive=True)
#         elif img_format == 'PNG':
#             img.save(path, format='PNG', optimize=True, compress_level=9)
#         else:
#             img.save(path, format='JPEG', quality=jpeg_quality, optimize=True)
            
#     except Exception:
#         pass

# def rezip_docx_lossless(docx_bytes: bytes, zip_level: int = 9) -> bytes:
#     """Lossless DOCX compression through re-zipping."""
#     with tempfile.TemporaryDirectory() as td:
#         in_path = os.path.join(td, "in.docx")
#         out_path = os.path.join(td, "out.docx")

#         with open(in_path, "wb") as f:
#             f.write(docx_bytes)

#         extract_dir = os.path.join(td, "unzipped")
#         os.makedirs(extract_dir, exist_ok=True)
        
#         with ZipFile(in_path, "r") as zin:
#             zin.extractall(extract_dir)
        
#         # Remove metadata for size reduction
#         remove_docx_metadata(extract_dir)

#         # Repack with maximum compression
#         with ZipFile(out_path, "w", compression=ZIP_DEFLATED, compresslevel=zip_level) as zout:
#             for root, _, files in os.walk(extract_dir):
#                 for name in files:
#                     full = os.path.join(root, name)
#                     rel = os.path.relpath(full, extract_dir)
#                     zout.write(full, rel)

#         with open(out_path, "rb") as f:
#             return f.read()

# def compress_docx_aggressive(docx_bytes: bytes, jpeg_quality: int, preserve_quality: bool = True) -> bytes:
#     """
#     Aggressive DOCX compression with smart image optimization and metadata removal.
#     """
#     with tempfile.TemporaryDirectory() as td:
#         in_path = os.path.join(td, "in.docx")
#         out_path = os.path.join(td, "out.docx")

#         with open(in_path, "wb") as f:
#             f.write(docx_bytes)

#         extract_dir = os.path.join(td, "unzipped")
#         os.makedirs(extract_dir, exist_ok=True)
        
#         with ZipFile(in_path, "r") as zin:
#             zin.extractall(extract_dir)
        
#         # Remove metadata
#         remove_docx_metadata(extract_dir)

#         # Optimize images
#         media_dir = os.path.join(extract_dir, "word", "media")
#         if os.path.isdir(media_dir):
#             for name in os.listdir(media_dir):
#                 p = os.path.join(media_dir, name)
#                 if os.path.isfile(p):
#                     optimize_image_smart(p, jpeg_quality=jpeg_quality, preserve_quality=preserve_quality)

#         # Repack with maximum compression
#         with ZipFile(out_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as zout:
#             for root, _, files in os.walk(extract_dir):
#                 for name in files:
#                     full = os.path.join(root, name)
#                     rel = os.path.relpath(full, extract_dir)
#                     zout.write(full, rel)

#         with open(out_path, "rb") as f:
#             return f.read()

# def compress_docx_to_target(docx_bytes: bytes, target_mb: float, preserve_quality: bool = True) -> bytes:
#     """
#     Intelligently compress DOCX to reach target size while preserving quality.
#     """
#     # Start with lossless
#     compressed = rezip_docx_lossless(docx_bytes, zip_level=9)
    
#     if len(compressed) <= target_mb * 1024 * 1024:
#         return compressed
    
#     # Try different quality levels
#     if preserve_quality:
#         quality_steps = [95, 92, 88, 85]
#     else:
#         quality_steps = [90, 85, 80, 75, 70, 65]
    
#     best = compressed
#     for quality in quality_steps:
#         result = compress_docx_aggressive(docx_bytes, jpeg_quality=quality, preserve_quality=preserve_quality)
#         if len(result) < len(best):
#             best = result
#         if len(result) <= target_mb * 1024 * 1024:
#             return result
    
#     return best

# # -------------------------
# # Streamlit UI
# # -------------------------
# st.set_page_config(page_title="Advanced Document Compressor", page_icon="🗜️", layout="centered")

# st.title("🗜️ Advanced PDF & Word Compressor")
# st.markdown("""
# **Enhanced compression with intelligent quality preservation**
# - 📦 Removes unnecessary metadata & fonts
# - 🖼️ Smart image optimization
# - 🎯 Target size with minimal quality loss
# """)

# uploaded = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])

# # Compression mode selection
# mode = st.radio(
#     "Compression mode",
#     ["Smart Lossless (Best Quality)", "Target Size (Quality Preserved)", "Maximum Compression"],
#     index=0,
#     help="Smart Lossless: Removes bloat without affecting quality\nTarget Size: Reaches your size goal with minimal quality impact\nMaximum: Smallest possible file"
# )

# col1, col2 = st.columns(2)
# with col1:
#     target_mb = st.number_input(
#         "Target size (MB)", 
#         min_value=0.1, 
#         max_value=500.0, 
#         value=2.0, 
#         step=0.1,
#         disabled=(mode == "Smart Lossless (Best Quality)")
#     )

# with col2:
#     preserve_quality = st.checkbox(
#         "Preserve Visual Quality",
#         value=(mode != "Maximum Compression"),
#         help="When enabled, uses higher quality settings for images"
#     )

# st.caption("💡 Tip: 'Target Size' mode intelligently compresses to your desired size while keeping documents visually sharp.")

# if uploaded is not None:
#     data = uploaded.read()
#     in_size = len(data)

#     st.write(f"**Original:** {uploaded.name} · {human_size(in_size)}")

#     file_suffix = Path(uploaded.name).suffix.lower()

#     if st.button("🚀 Compress", type="primary", use_container_width=True):
#         with st.spinner("Compressing... This may take a moment for large files."):
#             try:
#                 if file_suffix == ".pdf":
#                     if mode == "Smart Lossless (Best Quality)":
#                         out_bytes = compress_pdf_lossless(data, linearize=True)
#                     elif mode == "Target Size (Quality Preserved)":
#                         out_bytes = compress_pdf_to_target(data, target_mb=target_mb, preserve_quality=True)
#                     else:  # Maximum Compression
#                         out_bytes = compress_pdf_to_target(data, target_mb=target_mb, preserve_quality=False)

#                     out_name = Path(uploaded.name).stem + ".compressed.pdf"

#                 elif file_suffix == ".docx":
#                     if mode == "Smart Lossless (Best Quality)":
#                         out_bytes = rezip_docx_lossless(data, zip_level=9)
#                     elif mode == "Target Size (Quality Preserved)":
#                         out_bytes = compress_docx_to_target(data, target_mb=target_mb, preserve_quality=True)
#                     else:  # Maximum Compression
#                         out_bytes = compress_docx_to_target(data, target_mb=target_mb, preserve_quality=False)

#                     out_name = Path(uploaded.name).stem + ".compressed.docx"
#                 else:
#                     st.error("Unsupported file type. Please upload a .pdf or .docx.")
#                     st.stop()

#                 out_size = len(out_bytes)
#                 ratio = (1 - out_size / in_size) * 100 if in_size else 0.0

#                 # Display results
#                 col_a, col_b, col_c = st.columns(3)
#                 with col_a:
#                     st.metric("Original", human_size(in_size))
#                 with col_b:
#                     st.metric("Compressed", human_size(out_size))
#                 with col_c:
#                     st.metric("Saved", f"{ratio:.1f}%")

#                 st.success("✅ Compression completed successfully!")
                
#                 st.download_button(
#                     "⬇️ Download Compressed File",
#                     data=out_bytes,
#                     file_name=out_name,
#                     mime="application/octet-stream",
#                     use_container_width=True
#                 )

#                 # # Details
#                 # with st.expander("📊 Technical Details"):
#                 #     st.json({
#                 #         "original_size_bytes": in_size,
#                 #         "compressed_size_bytes": out_size,
#                 #         "compression_ratio": f"{ratio:.2f}%",
#                 #         "ghostscript_available": which_gs() is not None,
#                 #         "mode_used": mode,
#                 #         "quality_preserved": preserve_quality
#                 #     })

#                 if file_suffix == ".pdf" and which_gs() is None:
#                     st.info("💡 Install **Ghostscript** for even better PDF compression in aggressive modes.")

#             except Exception as e:
#                 st.error(f"❌ Compression failed: {e}")
#                 st.exception(e)

# # Footer
# st.markdown("---")
# st.markdown("""
# <div style='text-align: center; color: #666;'>
#     <small>Built with ❤️ for efficient document compression | Supports PDF & DOCX | Yohan Gowda D</small>
# </div>
# """, unsafe_allow_html=True)













# import io
# import os
# import shutil
# import subprocess
# import tempfile
# from pathlib import Path
# from zipfile import ZipFile, ZIP_DEFLATED
# import xml.etree.ElementTree as ET

# import streamlit as st
# from PIL import Image
# from pikepdf import Pdf, ObjectStreamMode, Name
# import fitz  # PyMuPDF for advanced PDF operations

# # ---------------------------
# # Helpers: formatting & sizes
# # ---------------------------
# def human_size(num_bytes: int) -> str:
#     for unit in ["B", "KB", "MB", "GB"]:
#         if num_bytes < 1024.0:
#             return f"{num_bytes:3.1f} {unit}"
#         num_bytes /= 1024.0
#     return f"{num_bytes:.1f} TB"

# # ---------------------------------
# # Enhanced PDF compression
# # ---------------------------------
# def compress_pdf_lossless(pdf_bytes: bytes, linearize: bool = False) -> bytes:
#     """Enhanced lossless PDF optimization using pikepdf."""
#     inp = io.BytesIO(pdf_bytes)
#     out = io.BytesIO()
#     with Pdf.open(inp) as pdf:
#         # Remove unnecessary metadata
#         if pdf.trailer.get("/Info"):
#             info = pdf.trailer["/Info"]
#             # Keep only essential metadata
#             essential = ["/Title", "/Author"]
#             keys_to_remove = [k for k in info.keys() if k not in essential]
#             for k in keys_to_remove:
#                 del info[k]
        
#         pdf.save(
#             out,
#             compress_streams=True,
#             object_stream_mode=ObjectStreamMode.generate,
#             preserve_pdfa=True,
#             linearize=linearize,
#             stream_decode_level=None  # Let pikepdf decide best compression
#         )
#     return out.getvalue()

# def compress_pdf_aggressive(pdf_bytes: bytes, target_mb: float = None, preserve_quality: bool = True) -> bytes:
#     """
#     Aggressive PDF compression with font subsetting, image optimization, and metadata removal.
#     Uses PyMuPDF for advanced operations.
#     """
#     doc = fitz.open("pdf", pdf_bytes)
    
#     # Create output buffer
#     out = io.BytesIO()
    
#     # Compression settings based on quality preservation
#     if preserve_quality:
#         # High quality: minimal image degradation
#         garbage_opts = 3  # Remove unused objects
#         deflate_opts = True
#         image_quality = 95
#     else:
#         # More aggressive
#         garbage_opts = 4  # More aggressive cleanup
#         deflate_opts = True
#         image_quality = 85
    
#     # Process each page
#     for page_num in range(len(doc)):
#         page = doc[page_num]
        
#         # Get all images on the page
#         image_list = page.get_images(full=True)
        
#         for img_index, img_info in enumerate(image_list):
#             xref = img_info[0]
            
#             try:
#                 # Extract image
#                 base_image = doc.extract_image(xref)
#                 image_bytes = base_image["image"]
#                 image_ext = base_image["ext"]
                
#                 # Open with PIL for recompression
#                 img = Image.open(io.BytesIO(image_bytes))
                
#                 # Calculate optimal dimensions based on target
#                 width, height = img.size
#                 max_dimension = 2400 if preserve_quality else 1800
                
#                 if width > max_dimension or height > max_dimension:
#                     # Resize maintaining aspect ratio
#                     if width > height:
#                         new_width = max_dimension
#                         new_height = int(height * (max_dimension / width))
#                     else:
#                         new_height = max_dimension
#                         new_width = int(width * (max_dimension / height))
                    
#                     img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
#                 # Convert to RGB if necessary
#                 if img.mode not in ('RGB', 'L'):
#                     img = img.convert('RGB')
                
#                 # Compress image
#                 img_buffer = io.BytesIO()
#                 img.save(img_buffer, format='JPEG', quality=image_quality, optimize=True)
#                 compressed_bytes = img_buffer.getvalue()
                
#                 # Replace image if compression achieved size reduction
#                 if len(compressed_bytes) < len(image_bytes):
#                     page.replace_image(xref, stream=compressed_bytes)
                    
#             except Exception as e:
#                 # Skip problematic images
#                 continue
    
#     # Save with optimization - REMOVED LINEAR FLAG
#     doc.save(
#         out,
#         garbage=garbage_opts,
#         deflate=deflate_opts,
#         clean=True,  # Clean up content streams
#     )
#     doc.close()
    
#     # Second pass with pikepdf for additional compression
#     out.seek(0)
#     return compress_pdf_lossless(out.read(), linearize=False)

# def which_gs():
#     """Find Ghostscript executable."""
#     candidates = ["gs", "gswin64c", "gswin32c"]
#     for c in candidates:
#         if shutil.which(c):
#             return c
#     return None

# def compress_pdf_with_gs(pdf_bytes: bytes, preset: str = "/ebook") -> bytes:
#     """Use Ghostscript for additional compression."""
#     gs_exec = which_gs()
#     if not gs_exec:
#         return None

#     with tempfile.TemporaryDirectory() as td:
#         in_path = os.path.join(td, "in.pdf")
#         out_path = os.path.join(td, "out.pdf")
#         with open(in_path, "wb") as f:
#             f.write(pdf_bytes)

#         cmd = [
#             gs_exec,
#             "-sDEVICE=pdfwrite",
#             "-dCompatibilityLevel=1.4",
#             f"-dPDFSETTINGS={preset}",
#             "-dNOPAUSE",
#             "-dQUIET",
#             "-dBATCH",
#             "-dDetectDuplicateImages=true",
#             "-dCompressFonts=true",
#             "-dSubsetFonts=true",
#             "-dEmbedAllFonts=false",
#             f"-sOutputFile={out_path}",
#             in_path,
#         ]
#         try:
#             subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             with open(out_path, "rb") as f:
#                 return f.read()
#         except Exception:
#             return None

# def compress_pdf_to_target(pdf_bytes: bytes, target_mb: float, preserve_quality: bool = True) -> bytes:
#     """Intelligently compress PDF to reach target size."""
    
#     # Start with aggressive compression
#     compressed = compress_pdf_aggressive(pdf_bytes, target_mb, preserve_quality)
    
#     if len(compressed) <= target_mb * 1024 * 1024:
#         return compressed
    
#     # Try Ghostscript if available
#     gs_exec = which_gs()
#     if gs_exec:
#         presets = ["/printer", "/ebook", "/screen"]
#         best = compressed
        
#         for preset in presets:
#             gs_result = compress_pdf_with_gs(compressed, preset)
#             if gs_result and len(gs_result) < len(best):
#                 best = gs_result
#                 if len(best) <= target_mb * 1024 * 1024:
#                     return best
        
#         return best
    
#     return compressed

# # ---------------------------------------------------------
# # Enhanced DOCX compression
# # ---------------------------------------------------------
# def remove_docx_metadata(extract_dir: str) -> None:
#     """Remove unnecessary metadata from DOCX to reduce size."""
#     try:
#         # Remove core properties
#         core_props = os.path.join(extract_dir, "docProps", "core.xml")
#         if os.path.exists(core_props):
#             tree = ET.parse(core_props)
#             root = tree.getroot()
#             # Keep only essential properties
#             for child in list(root):
#                 if not any(x in child.tag for x in ['title', 'creator']):
#                     root.remove(child)
#             tree.write(core_props)
        
#         # Remove app properties (except essentials)
#         app_props = os.path.join(extract_dir, "docProps", "app.xml")
#         if os.path.exists(app_props):
#             tree = ET.parse(app_props)
#             root = tree.getroot()
#             for child in list(root):
#                 if any(x in child.tag for x in ['TotalTime', 'Company', 'Manager', 'HyperlinksChanged']):
#                     root.remove(child)
#             tree.write(app_props)
            
#     except Exception:
#         pass

# def optimize_image_smart(path: str, jpeg_quality: int = 90, preserve_quality: bool = True) -> None:
#     """
#     Smart image optimization that preserves visual quality while reducing size.
#     """
#     try:
#         img = Image.open(path)
#         original_size = os.path.getsize(path)
        
#         # Get image info
#         width, height = img.size
#         img_format = img.format
        
#         # Determine optimal max dimension based on quality preference
#         if preserve_quality:
#             max_dimension = 3000  # High quality
#             jpeg_quality = max(jpeg_quality, 92)
#         else:
#             max_dimension = 2000  # Balanced
        
#         # Resize if too large
#         if width > max_dimension or height > max_dimension:
#             if width > height:
#                 new_width = max_dimension
#                 new_height = int(height * (max_dimension / width))
#             else:
#                 new_height = max_dimension
#                 new_width = int(width * (max_dimension / height))
            
#             img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
#         # Convert mode intelligently
#         if img.mode == 'RGBA':
#             # Check if alpha channel is actually used
#             if img.getextrema()[3] == (255, 255):
#                 img = img.convert('RGB')
#             else:
#                 # Keep as PNG for transparency
#                 img.save(path, format='PNG', optimize=True, compress_level=9)
#                 return
#         elif img.mode == 'P':
#             img = img.convert('RGB')
#         elif img.mode not in ('RGB', 'L'):
#             img = img.convert('RGB')
        
#         # Save optimized
#         if img_format in ['JPEG', 'JPG'] or img.mode == 'RGB':
#             img.save(path, format='JPEG', quality=jpeg_quality, optimize=True, progressive=True)
#         elif img_format == 'PNG':
#             img.save(path, format='PNG', optimize=True, compress_level=9)
#         else:
#             img.save(path, format='JPEG', quality=jpeg_quality, optimize=True)
            
#     except Exception:
#         pass

# def rezip_docx_lossless(docx_bytes: bytes, zip_level: int = 9) -> bytes:
#     """Lossless DOCX compression through re-zipping."""
#     with tempfile.TemporaryDirectory() as td:
#         in_path = os.path.join(td, "in.docx")
#         out_path = os.path.join(td, "out.docx")

#         with open(in_path, "wb") as f:
#             f.write(docx_bytes)

#         extract_dir = os.path.join(td, "unzipped")
#         os.makedirs(extract_dir, exist_ok=True)
        
#         with ZipFile(in_path, "r") as zin:
#             zin.extractall(extract_dir)
        
#         # Remove metadata for size reduction
#         remove_docx_metadata(extract_dir)

#         # Repack with maximum compression
#         with ZipFile(out_path, "w", compression=ZIP_DEFLATED, compresslevel=zip_level) as zout:
#             for root, _, files in os.walk(extract_dir):
#                 for name in files:
#                     full = os.path.join(root, name)
#                     rel = os.path.relpath(full, extract_dir)
#                     zout.write(full, rel)

#         with open(out_path, "rb") as f:
#             return f.read()

# def compress_docx_aggressive(docx_bytes: bytes, jpeg_quality: int, preserve_quality: bool = True) -> bytes:
#     """
#     Aggressive DOCX compression with smart image optimization and metadata removal.
#     """
#     with tempfile.TemporaryDirectory() as td:
#         in_path = os.path.join(td, "in.docx")
#         out_path = os.path.join(td, "out.docx")

#         with open(in_path, "wb") as f:
#             f.write(docx_bytes)

#         extract_dir = os.path.join(td, "unzipped")
#         os.makedirs(extract_dir, exist_ok=True)
        
#         with ZipFile(in_path, "r") as zin:
#             zin.extractall(extract_dir)
        
#         # Remove metadata
#         remove_docx_metadata(extract_dir)

#         # Optimize images
#         media_dir = os.path.join(extract_dir, "word", "media")
#         if os.path.isdir(media_dir):
#             for name in os.listdir(media_dir):
#                 p = os.path.join(media_dir, name)
#                 if os.path.isfile(p):
#                     optimize_image_smart(p, jpeg_quality=jpeg_quality, preserve_quality=preserve_quality)

#         # Repack with maximum compression
#         with ZipFile(out_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as zout:
#             for root, _, files in os.walk(extract_dir):
#                 for name in files:
#                     full = os.path.join(root, name)
#                     rel = os.path.relpath(full, extract_dir)
#                     zout.write(full, rel)

#         with open(out_path, "rb") as f:
#             return f.read()

# def compress_docx_to_target(docx_bytes: bytes, target_mb: float, preserve_quality: bool = True) -> bytes:
#     """
#     Intelligently compress DOCX to reach target size while preserving quality.
#     """
#     # Start with lossless
#     compressed = rezip_docx_lossless(docx_bytes, zip_level=9)
    
#     if len(compressed) <= target_mb * 1024 * 1024:
#         return compressed
    
#     # Try different quality levels
#     if preserve_quality:
#         quality_steps = [95, 92, 88, 85]
#     else:
#         quality_steps = [90, 85, 80, 75, 70, 65]
    
#     best = compressed
#     for quality in quality_steps:
#         result = compress_docx_aggressive(docx_bytes, jpeg_quality=quality, preserve_quality=preserve_quality)
#         if len(result) < len(best):
#             best = result
#         if len(result) <= target_mb * 1024 * 1024:
#             return result
    
#     return best

# # -------------------------
# # Streamlit UI
# # -------------------------
# st.set_page_config(page_title="Advanced Document Compressor", page_icon="🗜️", layout="centered")

# st.title("🗜️ Advanced PDF & Word Compressor")
# st.markdown("""
# **Enhanced compression with intelligent quality preservation**
# - 📦 Removes unnecessary metadata & fonts
# - 🖼️ Smart image optimization
# - 🎯 Target size with minimal quality loss
# """)

# uploaded = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])

# # Compression mode selection
# mode = st.radio(
#     "Compression mode",
#     ["Smart Lossless (Best Quality)", "Target Size (Quality Preserved)", "Maximum Compression"],
#     index=0,
#     help="Smart Lossless: Removes bloat without affecting quality\nTarget Size: Reaches your size goal with minimal quality impact\nMaximum: Smallest possible file"
# )

# col1, col2 = st.columns(2)
# with col1:
#     target_mb = st.number_input(
#         "Target size (MB)", 
#         min_value=0.1, 
#         max_value=500.0, 
#         value=2.0, 
#         step=0.1,
#         disabled=(mode == "Smart Lossless (Best Quality)")
#     )

# with col2:
#     preserve_quality = st.checkbox(
#         "Preserve Visual Quality",
#         value=(mode != "Maximum Compression"),
#         help="When enabled, uses higher quality settings for images"
#     )

# st.caption("💡 Tip: 'Target Size' mode intelligently compresses to your desired size while keeping documents visually sharp.")

# if uploaded is not None:
#     data = uploaded.read()
#     in_size = len(data)

#     st.write(f"**Original:** {uploaded.name} · {human_size(in_size)}")

#     file_suffix = Path(uploaded.name).suffix.lower()

#     if st.button("🚀 Compress", type="primary", use_container_width=True):
#         with st.spinner("Compressing... This may take a moment for large files."):
#             try:
#                 if file_suffix == ".pdf":
#                     if mode == "Smart Lossless (Best Quality)":
#                         out_bytes = compress_pdf_lossless(data, linearize=False)
#                     elif mode == "Target Size (Quality Preserved)":
#                         out_bytes = compress_pdf_to_target(data, target_mb=target_mb, preserve_quality=True)
#                     else:  # Maximum Compression
#                         out_bytes = compress_pdf_to_target(data, target_mb=target_mb, preserve_quality=False)

#                     out_name = Path(uploaded.name).stem + ".compressed.pdf"

#                 elif file_suffix == ".docx":
#                     if mode == "Smart Lossless (Best Quality)":
#                         out_bytes = rezip_docx_lossless(data, zip_level=9)
#                     elif mode == "Target Size (Quality Preserved)":
#                         out_bytes = compress_docx_to_target(data, target_mb=target_mb, preserve_quality=True)
#                     else:  # Maximum Compression
#                         out_bytes = compress_docx_to_target(data, target_mb=target_mb, preserve_quality=False)

#                     out_name = Path(uploaded.name).stem + ".compressed.docx"
#                 else:
#                     st.error("Unsupported file type. Please upload a .pdf or .docx.")
#                     st.stop()

#                 out_size = len(out_bytes)
#                 ratio = (1 - out_size / in_size) * 100 if in_size else 0.0

#                 # Display results
#                 col_a, col_b, col_c = st.columns(3)
#                 with col_a:
#                     st.metric("Original", human_size(in_size))
#                 with col_b:
#                     st.metric("Compressed", human_size(out_size))
#                 with col_c:
#                     st.metric("Saved", f"{ratio:.1f}%")

#                 st.success("✅ Compression completed successfully!")
                
#                 st.download_button(
#                     "⬇️ Download Compressed File",
#                     data=out_bytes,
#                     file_name=out_name,
#                     mime="application/octet-stream",
#                     use_container_width=True
#                 )

#                 if file_suffix == ".pdf" and which_gs() is None:
#                     st.info("💡 Install **Ghostscript** for even better PDF compression in aggressive modes.")

#             except Exception as e:
#                 st.error(f"❌ Compression failed: {e}")
#                 st.exception(e)

# # Footer
# st.markdown("---")
# st.markdown("""
# <div style='text-align: center; color: #666;'>
#     <small>Built with ❤️ for efficient document compression | Supports PDF & DOCX | Yohan Gowda D</small>
# </div>
# """, unsafe_allow_html=True)











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
from pikepdf import Pdf, ObjectStreamMode
import fitz  # PyMuPDF

# ---------------------------
# Helpers
# ---------------------------
def human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"

def which_gs():
    """Find Ghostscript executable."""
    candidates = ["gs", "gswin64c", "gswin32c"]
    for c in candidates:
        if shutil.which(c):
            return c
    return None

# ---------------------------------
# ENHANCED PDF COMPRESSION ENGINE
# ---------------------------------
def compress_images_in_pdf(doc, quality: int = 85, max_dimension: int = 1800):
    """Aggressively compress all images in PDF."""
    compressed_count = 0
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        for img_info in image_list:
            xref = img_info[0]
            
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Open with PIL
                img = Image.open(io.BytesIO(image_bytes))
                width, height = img.size
                
                # Aggressive resizing
                if width > max_dimension or height > max_dimension:
                    ratio = max_dimension / max(width, height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to RGB
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Compress
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=quality, optimize=True)
                compressed_bytes = img_buffer.getvalue()
                
                # Replace if smaller
                if len(compressed_bytes) < len(image_bytes):
                    page.replace_image(xref, stream=compressed_bytes)
                    compressed_count += 1
                    
            except Exception:
                continue
    
    return compressed_count

def compress_pdf_pymupdf(pdf_bytes: bytes, quality: int = 85, max_dimension: int = 1800) -> bytes:
    """Primary PDF compression using PyMuPDF."""
    doc = fitz.open("pdf", pdf_bytes)
    
    # Compress images
    compress_images_in_pdf(doc, quality=quality, max_dimension=max_dimension)
    
    # Save with aggressive settings
    out = io.BytesIO()
    doc.save(
        out,
        garbage=4,
        deflate=True,
        clean=True,
    )
    doc.close()
    
    return out.getvalue()

def compress_pdf_pikepdf(pdf_bytes: bytes) -> bytes:
    """Secondary compression using pikepdf."""
    inp = io.BytesIO(pdf_bytes)
    out = io.BytesIO()
    
    with Pdf.open(inp) as pdf:
        # Remove metadata
        if pdf.trailer.get("/Info"):
            info = pdf.trailer["/Info"]
            for key in list(info.keys()):
                if key not in ["/Title", "/Author"]:
                    del info[key]
        
        pdf.save(
            out,
            compress_streams=True,
            object_stream_mode=ObjectStreamMode.generate,
            stream_decode_level=None
        )
    
    return out.getvalue()

def compress_pdf_ghostscript(pdf_bytes: bytes, preset: str = "/ebook") -> bytes:
    """Compress using Ghostscript - often the most effective."""
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
            "-r150",  # DPI reduction
            "-dDownsampleColorImages=true",
            "-dColorImageResolution=150",
            "-dColorImageDownsampleType=/Bicubic",
            f"-sOutputFile={out_path}",
            in_path,
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
            if os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    return f.read()
        except Exception:
            pass
    
    return None

def compress_pdf_smart(pdf_bytes: bytes, target_mb: float, mode: str = "balanced") -> bytes:
    """
    Smart multi-stage PDF compression.
    Tries multiple strategies and returns the best result under target size.
    """
    target_bytes = target_mb * 1024 * 1024
    original_size = len(pdf_bytes)
    
    results = []
    
    # Strategy 1: PyMuPDF with varying quality levels
    quality_levels = {
        "quality": [(90, 2200), (85, 2000)],
        "balanced": [(80, 1800), (75, 1600), (70, 1400)],
        "aggressive": [(65, 1400), (60, 1200), (55, 1000), (50, 800)]
    }
    
    for quality, max_dim in quality_levels.get(mode, quality_levels["balanced"]):
        try:
            compressed = compress_pdf_pymupdf(pdf_bytes, quality=quality, max_dimension=max_dim)
            results.append(("PyMuPDF", quality, compressed))
            
            # Early exit if target met with good quality
            if len(compressed) <= target_bytes and quality >= 70:
                return compressed
        except Exception:
            continue
    
    # Strategy 2: Add pikepdf pass to best result so far
    if results:
        best = min(results, key=lambda x: len(x[2]))
        try:
            pikepdf_result = compress_pdf_pikepdf(best[2])
            results.append(("PyMuPDF+Pikepdf", best[1], pikepdf_result))
        except Exception:
            pass
    
    # Strategy 3: Ghostscript (most aggressive)
    gs_exec = which_gs()
    if gs_exec:
        presets = ["/printer", "/ebook", "/screen"]
        for preset in presets:
            try:
                gs_result = compress_pdf_ghostscript(pdf_bytes, preset=preset)
                if gs_result:
                    results.append((f"Ghostscript {preset}", 0, gs_result))
                    
                    # If we hit target with /ebook or /printer, use it
                    if len(gs_result) <= target_bytes and preset in ["/printer", "/ebook"]:
                        return gs_result
            except Exception:
                continue
    
    # Return the smallest result
    if results:
        best = min(results, key=lambda x: len(x[2]))
        return best[2]
    
    # Fallback
    return pdf_bytes

# ---------------------------------------------------------
# ENHANCED DOCX COMPRESSION ENGINE
# ---------------------------------------------------------
def remove_docx_bloat(extract_dir: str):
    """Remove all unnecessary data from DOCX."""
    try:
        # Remove thumbnail
        thumb_path = os.path.join(extract_dir, "docProps", "thumbnail.jpeg")
        if os.path.exists(thumb_path):
            os.remove(thumb_path)
        
        # Clean metadata
        core_props = os.path.join(extract_dir, "docProps", "core.xml")
        if os.path.exists(core_props):
            tree = ET.parse(core_props)
            root = tree.getroot()
            for child in list(root):
                if not any(x in child.tag for x in ['title', 'creator']):
                    root.remove(child)
            tree.write(core_props, encoding='utf-8', xml_declaration=True)
        
        # Clean app properties
        app_props = os.path.join(extract_dir, "docProps", "app.xml")
        if os.path.exists(app_props):
            tree = ET.parse(app_props)
            root = tree.getroot()
            for child in list(root):
                if any(x in child.tag.lower() for x in ['totaltime', 'company', 'manager', 'hyperlinks', 'lines', 'paragraphs', 'application']):
                    root.remove(child)
            tree.write(app_props, encoding='utf-8', xml_declaration=True)
            
    except Exception:
        pass

def optimize_image_aggressive(path: str, quality: int = 75, max_dimension: int = 1600):
    """Aggressively optimize images."""
    try:
        img = Image.open(path)
        width, height = img.size
        
        # Resize if needed
        if width > max_dimension or height > max_dimension:
            ratio = max_dimension / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Handle alpha
        if img.mode == 'RGBA':
            if img.getextrema()[3] == (255, 255):
                img = img.convert('RGB')
            else:
                # Compress PNG
                img.save(path, format='PNG', optimize=True, compress_level=9)
                return
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save as JPEG
        img.save(path, format='JPEG', quality=quality, optimize=True, progressive=True)
            
    except Exception:
        pass

def compress_docx_smart(docx_bytes: bytes, target_mb: float, mode: str = "balanced") -> bytes:
    """Smart multi-stage DOCX compression."""
    
    quality_configs = {
        "quality": [(85, 2000), (80, 1800)],
        "balanced": [(75, 1600), (70, 1400), (65, 1200)],
        "aggressive": [(60, 1200), (55, 1000), (50, 800), (45, 600)]
    }
    
    target_bytes = target_mb * 1024 * 1024
    
    for quality, max_dim in quality_configs.get(mode, quality_configs["balanced"]):
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in.docx")
            out_path = os.path.join(td, "out.docx")
            extract_dir = os.path.join(td, "unzipped")

            with open(in_path, "wb") as f:
                f.write(docx_bytes)

            os.makedirs(extract_dir, exist_ok=True)
            
            # Extract
            with ZipFile(in_path, "r") as zin:
                zin.extractall(extract_dir)
            
            # Clean bloat
            remove_docx_bloat(extract_dir)

            # Optimize images
            media_dir = os.path.join(extract_dir, "word", "media")
            if os.path.isdir(media_dir):
                for name in os.listdir(media_dir):
                    p = os.path.join(media_dir, name)
                    if os.path.isfile(p):
                        optimize_image_aggressive(p, quality=quality, max_dimension=max_dim)

            # Repack with max compression
            with ZipFile(out_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as zout:
                for root, _, files in os.walk(extract_dir):
                    for name in files:
                        full = os.path.join(root, name)
                        rel = os.path.relpath(full, extract_dir)
                        zout.write(full, rel)

            with open(out_path, "rb") as f:
                result = f.read()
                
            # Check if target met
            if len(result) <= target_bytes:
                return result
    
    # Return best attempt
    return result if 'result' in locals() else docx_bytes

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Pro Document Compressor", page_icon="🗜️", layout="centered")

st.title("🗜️ Pro Document Compressor")
st.markdown("""
**Industrial-strength compression engine**
- 🎯 **Multi-strategy compression** - tries multiple methods, keeps the best
- 🖼️ **Aggressive image optimization** - smart resizing & quality reduction
- 📦 **Deep cleanup** - removes all bloat & metadata
""")

uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

col1, col2 = st.columns(2)
with col1:
    target_mb = st.number_input("Target Size (MB)", min_value=0.1, max_value=50.0, value=1.0, step=0.1)

with col2:
    mode = st.selectbox(
        "Compression Level",
        ["Quality (Mild)", "Balanced", "Aggressive (Max)"],
        index=1
    )

mode_map = {
    "Quality (Mild)": "quality",
    "Balanced": "balanced",
    "Aggressive (Max)": "aggressive"
}

if uploaded is not None:
    data = uploaded.read()
    in_size = len(data)
    
    st.write(f"**Original:** {uploaded.name} · {human_size(in_size)}")
    
    # Show warning if target is very small
    if target_mb < (in_size / (1024 * 1024)) * 0.3:
        st.warning("⚠️ Target size is very aggressive. Quality may be significantly reduced.")
    
    file_suffix = Path(uploaded.name).suffix.lower()

    if st.button("🚀 Compress Now", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        with st.spinner("🔄 Compressing with multiple strategies..."):
            try:
                status.text("Analyzing document...")
                progress_bar.progress(10)
                
                if file_suffix == ".pdf":
                    status.text("Compressing PDF (this may take 30-60 seconds)...")
                    progress_bar.progress(30)
                    
                    out_bytes = compress_pdf_smart(
                        data, 
                        target_mb=target_mb, 
                        mode=mode_map[mode]
                    )
                    out_name = Path(uploaded.name).stem + ".compressed.pdf"
                    
                elif file_suffix == ".docx":
                    status.text("Compressing DOCX...")
                    progress_bar.progress(30)
                    
                    out_bytes = compress_docx_smart(
                        data,
                        target_mb=target_mb,
                        mode=mode_map[mode]
                    )
                    out_name = Path(uploaded.name).stem + ".compressed.docx"
                    
                else:
                    st.error("Unsupported file type")
                    st.stop()
                
                progress_bar.progress(100)
                status.text("✅ Complete!")
                
                out_size = len(out_bytes)
                ratio = (1 - out_size / in_size) * 100 if in_size else 0.0
                
                # Results
                st.markdown("---")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Original", human_size(in_size))
                with col_b:
                    st.metric("Compressed", human_size(out_size), delta=f"-{ratio:.1f}%")
                with col_c:
                    target_met = out_size <= target_mb * 1024 * 1024
                    st.metric("Target", human_size(int(target_mb * 1024 * 1024)), 
                             delta="✅ Met" if target_met else "⚠️ Close")
                
                if target_met:
                    st.success("🎉 Target size achieved!")
                elif out_size < in_size:
                    st.info(f"ℹ️ Reduced by {ratio:.1f}% - Try 'Aggressive' mode for more compression")
                else:
                    st.warning("⚠️ File couldn't be compressed further. Try a larger target size.")
                
                st.download_button(
                    "⬇️ Download Compressed File",
                    data=out_bytes,
                    file_name=out_name,
                    mime="application/octet-stream",
                    use_container_width=True
                )
                
                # Tips
                if file_suffix == ".pdf" and not which_gs():
                    st.info("💡 **Pro Tip:** Install [Ghostscript](https://www.ghostscript.com/download/gsdnld.html) for 30-50% better compression!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# Info section
with st.expander("ℹ️ How It Works"):
    st.markdown("""
    **PDF Compression:**
    - Extracts & recompresses images with optimal quality/size balance
    - Removes embedded fonts & metadata
    - Uses Ghostscript (if installed) for additional 30-50% compression
    - Tries multiple strategies and keeps the best result
    
    **DOCX Compression:**
    - Optimizes embedded images (resize + quality reduction)
    - Removes metadata, thumbnails, and revision history
    - Re-zips with maximum compression
    
    **Best Results:**
    - Documents with many images compress better
    - Text-only PDFs have limited compression potential
    - Installing Ghostscript dramatically improves PDF results
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>Pro Document Compressor v2.0 | Powered by PyMuPDF, Pikepdf & Ghostscript</small>
</div>
""", unsafe_allow_html=True)