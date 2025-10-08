from flask import Flask, request, jsonify, send_file
import os
import cv2
import numpy as np
import base64
import io
from PIL import Image, ImageEnhance, ImageFilter
import warnings
import fitz  # PyMuPDF for PDF processing
import re
from typing import List, Tuple, Dict
import math
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import tempfile
warnings.filterwarnings('ignore')

# TrOCR imports
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
    print("✓ TrOCR imports successful")
except ImportError as e:
    print(f"✗ TrOCR import failed: {e}")
    TROCR_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

class HandwritingToTextConverter:
    def __init__(self):
        self.trocr_loaded = False
        if TROCR_AVAILABLE:
            try:
                print("Loading TrOCR models...")
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Using device: {self.device}")
                
                # Load both models for best results
                print("Loading handwritten text model...")
                self.processor_hw = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
                self.model_hw = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(self.device)
                
                print("Loading printed text model...")
                self.processor_pr = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
                self.model_pr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').to(self.device)
                
                self.trocr_loaded = True
                print("✓ TrOCR models loaded successfully!")
            except Exception as e:
                print(f"✗ Error loading TrOCR: {e}")
                import traceback
                traceback.print_exc()
                self.trocr_loaded = False

    def is_image_file(self, filename: str) -> bool:
        """Check if the file is an image format"""
        if not filename:
            return False
        image_extensions = ['.png', '.jpg', '.jpeg']
        filename_lower = filename.lower()
        result = any(filename_lower.endswith(ext) for ext in image_extensions)
        print(f"DEBUG: Python is_image_file('{filename}') = {result}")
        print(f"  - Filename lower: '{filename_lower}'")
        print(f"  - Checked extensions: {image_extensions}")
        return result

    def is_pdf_file(self, filename: str) -> bool:
        """Check if the file is a PDF"""
        if not filename:
            return False
        filename_lower = filename.lower()
        result = filename_lower.endswith('.pdf')
        print(f"DEBUG: Python is_pdf_file('{filename}') = {result}")
        print(f"  - Filename lower: '{filename_lower}'")
        return result

    def load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Load image from bytes and convert to numpy array"""
        try:
            print(f"Loading image from {len(image_bytes)} bytes...")
            
            # Validate image bytes
            if len(image_bytes) == 0:
                print("✗ Image bytes are empty")
                return None
                
            # Check for common image signatures
            if image_bytes[:4] == b'\x89PNG':
                print("✓ Detected PNG image")
            elif image_bytes[:2] == b'\xff\xd8':
                print("✓ Detected JPEG image")
            else:
                print("⚠ Warning: Unknown image format, but proceeding...")
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            print(f"PIL Image loaded: {image.mode}, size: {image.size}")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print("Converted image to RGB")
            
            # Convert to numpy array
            img_array = np.array(image)
            print(f"Numpy array shape: {img_array.shape}")
            
            return img_array
        except Exception as e:
            print(f"✗ Error loading image from bytes: {e}")
            import traceback
            traceback.print_exc()
            return None

    def pdf_to_images(self, pdf_bytes: bytes, dpi: int = 350) -> List[np.ndarray]:
        """Convert PDF pages to high-resolution images"""
        images = []
        try:
            print(f"Converting PDF ({len(pdf_bytes)} bytes) to images at {dpi} DPI...")
            
            # Validate PDF bytes
            if len(pdf_bytes) == 0:
                print("✗ PDF bytes are empty")
                return []
                
            if not pdf_bytes.startswith(b'%PDF'):
                print("⚠ Warning: PDF doesn't start with expected signature")
            
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            print(f"PDF opened successfully, {len(pdf_document)} pages")
            
            for page_num in range(len(pdf_document)):
                print(f"Processing PDF page {page_num + 1}...")
                page = pdf_document[page_num]
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    print(f"✓ PDF page {page_num + 1} converted to image shape: {img.shape}")
                else:
                    print(f"✗ Failed to decode page {page_num + 1}")
                
            pdf_document.close()
            print(f"Successfully converted {len(images)} pages from PDF")
            return images
        except Exception as e:
            print(f"✗ Error converting PDF to images: {e}")
            import traceback
            traceback.print_exc()
            return []

    def advanced_image_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """Minimal preprocessing to preserve handwriting quality"""
        if img is None:
            return None
            
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        denoised = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced

    def detect_actual_text_regions(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Robust text region detection with improved line separation"""
        if img is None:
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape
        
        print(f"Image dimensions: {w}x{h}")
        
        binary_images = []
        
        _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_images.append(binary1)
        
        binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 21, 8)
        binary_images.append(binary2)
        
        mean_brightness = np.mean(gray)
        threshold_val = max(120, min(180, mean_brightness - 30))
        _, binary3 = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
        binary_images.append(binary3)
        
        combined_binary = np.zeros_like(binary1)
        for y in range(h):
            for x in range(w):
                votes = sum(1 for binary in binary_images if binary[y, x] > 0)
                if votes >= 2:
                    combined_binary[y, x] = 255
        
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, kernel_clean)
        
        contours, _ = cv2.findContours(combined_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        min_contour_area = 25
        min_width = 10
        min_height = 5
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_contour_area:
                x, y, w_c, h_c = cv2.boundingRect(contour)
                if w_c >= min_width and h_c >= min_height:
                    valid_contours.append((x, y, x + w_c, y + h_c))
        
        print(f"Found {len(valid_contours)} valid contours")
        
        if valid_contours:
            text_regions = self._group_contours_into_lines_improved(valid_contours, h, w)
        
        validated_regions = []
        for region in text_regions:
            if self._validate_text_region(combined_binary, region, w, h):
                validated_regions.append(region)
        
        print(f"Final validated regions: {len(validated_regions)}")
        return validated_regions

    def _group_contours_into_lines_improved(self, contours, img_h, img_w):
        """Ultra-sensitive grouping for very close lines with horizontal deviations"""
        if not contours:
            return []
        
        contours = sorted(contours, key=lambda x: x[1])
        
        lines = []
        current_line_contours = [contours[0]]
        
        for i in range(1, len(contours)):
            curr_contour = contours[i]
            prev_contour = contours[i-1]
            
            should_separate = self._should_separate_lines_ultra_sensitive(prev_contour, curr_contour, current_line_contours)
            
            if should_separate:
                if current_line_contours:
                    line_bbox = self._merge_contours_to_line(current_line_contours, img_w, img_h)
                    if line_bbox:
                        lines.append(line_bbox)
                
                current_line_contours = [curr_contour]
            else:
                current_line_contours.append(curr_contour)
        
        if current_line_contours:
            line_bbox = self._merge_contours_to_line(current_line_contours, img_w, img_h)
            if line_bbox:
                lines.append(line_bbox)
        
        return lines

    def _should_separate_lines_ultra_sensitive(self, prev_contour, curr_contour, current_line_contours):
        """Ultra-sensitive logic specifically for handwritten text with small gaps"""
        
        if len(current_line_contours) > 1:
            line_top = min(c[1] for c in current_line_contours)
            line_bottom = max(c[3] for c in current_line_contours)
            line_height = line_bottom - line_top
        else:
            line_top = current_line_contours[0][1]
            line_bottom = current_line_contours[0][3]
            line_height = line_bottom - line_top
        
        curr_top = curr_contour[1]
        curr_bottom = curr_contour[3]
        curr_height = curr_bottom - curr_top
        
        vertical_gap = curr_top - line_bottom
        
        overlap_start = max(line_top, curr_top)
        overlap_end = min(line_bottom, curr_bottom)
        overlap_height = max(0, overlap_end - overlap_start)
        
        if vertical_gap > 0 and overlap_height == 0:
            print(f"  Separating: vertical gap {vertical_gap}px with no overlap")
            return True
        
        if vertical_gap >= 0 and overlap_height > 0:
            min_height = min(line_height, curr_height)
            overlap_ratio = overlap_height / min_height if min_height > 0 else 0
            
            if overlap_ratio < 0.3:
                print(f"  Separating: gap {vertical_gap}px with small overlap ratio {overlap_ratio:.2f}")
                return True
        
        line_left = min(c[0] for c in current_line_contours)
        line_right = max(c[2] for c in current_line_contours)
        line_center_x = (line_left + line_right) / 2
        line_width = line_right - line_left
        
        curr_left = curr_contour[0]
        curr_right = curr_contour[2]
        curr_center_x = (curr_left + curr_right) / 2
        curr_width = curr_right - curr_left
        
        horizontal_deviation = abs(curr_center_x - line_center_x)
        
        max_width = max(line_width, curr_width)
        if max_width > 0:
            deviation_ratio = horizontal_deviation / max_width
            
            if deviation_ratio > 0.4 and vertical_gap >= 0:
                print(f"  Separating: horizontal deviation {deviation_ratio:.2f} with gap {vertical_gap}px")
                return True
        
        height_ratio = max(line_height, curr_height) / max(1, min(line_height, curr_height))
        if height_ratio > 2.5 and vertical_gap >= 0:
            print(f"  Separating: height ratio {height_ratio:.2f} with gap {vertical_gap}px")
            return True
        
        avg_height = (line_height + curr_height) / 2
        if curr_height < avg_height * 0.6 and vertical_gap > 0:
            print(f"  Separating: small contour {curr_height}px vs avg {avg_height:.1f}px with gap {vertical_gap}px")
            return True
        
        if (curr_right < line_left - 10) or (curr_left > line_right + 10):
            if vertical_gap >= -5:
                print(f"  Separating: horizontally separate regions with gap {vertical_gap}px")
                return True
        
        return False

    def _merge_contours_to_line(self, contours, img_w, img_h):
        """More lenient merging for short lines"""
        if not contours:
            return None
        
        min_x = min(c[0] for c in contours)
        min_y = min(c[1] for c in contours)
        max_x = max(c[2] for c in contours)
        max_y = max(c[3] for c in contours)
        
        padding_x = 8
        padding_y = 4
        
        x1 = max(0, min_x - padding_x)
        y1 = max(0, min_y - padding_y)
        x2 = min(img_w, max_x + padding_x)
        y2 = min(img_h, max_y + padding_y)
        
        width = x2 - x1
        height = y2 - y1
        
        if width < 15 or height < 6:
            return None
        
        return (x1, y1, x2, y2)

    def _validate_text_region(self, binary_img, region, img_w, img_h):
        """More lenient validation for short text regions"""
        x1, y1, x2, y2 = region
        
        region_img = binary_img[y1:y2, x1:x2]
        
        if region_img.size == 0:
            return False
        
        white_pixels = np.sum(region_img > 0)
        total_pixels = region_img.size
        text_density = white_pixels / total_pixels
        
        if text_density < 0.015:
            return False
        
        height, width = region_img.shape
        horizontal_projection = np.sum(region_img, axis=1) / 255
        
        non_zero_rows = np.sum(horizontal_projection > 0)
        if non_zero_rows < height * 0.2:
            return False
        
        aspect_ratio = width / height
        if aspect_ratio < 0.8:
            return False
        
        return True

    def preprocess_line_for_ocr(self, img: np.ndarray, line_box: Tuple[int, int, int, int]) -> np.ndarray:
        """Minimal preprocessing to avoid introducing artifacts"""
        x1, y1, x2, y2 = line_box
        
        line_img = img[y1:y2, x1:x2].copy()
        
        if line_img.size == 0:
            return None
        
        if len(line_img.shape) == 3:
            line_img = cv2.cvtColor(line_img, cv2.COLOR_RGB2GRAY)
        
        if np.mean(line_img) > 240:
            return None
        
        line_img = cv2.normalize(line_img, None, 0, 255, cv2.NORM_MINMAX)
        
        height, width = line_img.shape
        
        if height < 20:
            target_height = 32
        elif height > 120:
            target_height = 80
        else:
            target_height = height
        
        if height != target_height:
            scale = target_height / height
            new_width = max(int(width * scale), 64)
            new_width = min(new_width, 800)
            line_img = cv2.resize(line_img, (new_width, target_height), 
                                interpolation=cv2.INTER_CUBIC)
        
        pad_h, pad_w = 8, 12
        line_img = cv2.copyMakeBorder(line_img, pad_h, pad_h, pad_w, pad_w, 
                                    cv2.BORDER_CONSTANT, value=255)
        
        if np.mean(line_img) < 128:
            line_img = cv2.bitwise_not(line_img)
        
        line_img_rgb = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
        
        return line_img_rgb

    def extract_text_single_line(self, line_img, line_number, model_type='handwritten'):
        """Process a single line image with proper line numbering"""
        if line_img is None or not self.trocr_loaded:
            return "", 0.0
        
        processor = self.processor_hw if model_type == 'handwritten' else self.processor_pr
        model = self.model_hw if model_type == 'handwritten' else self.model_pr
        
        try:
            pil_img = Image.fromarray(line_img)
            
            pixel_values = processor(pil_img, return_tensors="pt").pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values,
                    max_length=80,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=False,
                    length_penalty=1.2,
                    repetition_penalty=1.2
                )
            
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            text = self.clean_extracted_text(text)
            
            if text.strip() and len(text.strip()) >= 2:
                confidence = self.calculate_text_confidence(text)
                
                if confidence > 0.15:
                    print(f"  Line {line_number}: '{text}' (conf: {confidence:.2f})")
                    return text, confidence
                else:
                    if self._has_recognizable_patterns(text):
                        print(f"  Line {line_number}: '{text}' (accepted by pattern recognition)")
                        return text, 0.25
                    else:
                        print(f"  Line {line_number}: Rejected low confidence: '{text}' (conf: {confidence:.2f})")
                        return "", 0.0
            else:
                if text.strip() and (text.strip().isalnum() or text.strip() in '.,!?;:()-'):
                    print(f"  Line {line_number}: Single char accepted: '{text}'")
                    return text, 0.2
                else:
                    print(f"  Line {line_number}: Rejected short text: '{text}'")
                    return "", 0.0
                    
        except Exception as e:
            print(f"Error processing line {line_number}: {e}")
            return "", 0.0

    def _has_recognizable_patterns(self, text: str) -> bool:
        """New method to identify potentially valid text even with low confidence"""
        if not text or len(text.strip()) < 2:
            return False
        
        text_clean = text.strip().lower()
        
        patterns = [
            r'^q\d+',
            r'^\(\s*[ivx]+\s*\)',
            r'^\d+[\.\):]',
            r'^[a-z]+\s+[a-z]+',
            r'\d+',
            r'[aeiou]',
        ]
        
        for pattern in patterns:
            if re.search(pattern, text_clean):
                return True
        
        alpha_chars = sum(1 for c in text_clean if c.isalpha())
        if alpha_chars >= 2:
            return True
        
        return False

    def clean_extracted_text(self, text: str) -> str:
        """Much more lenient text cleaning"""
        if not text:
            return ""
        
        text = text.strip()
        text = ' '.join(text.split())
        
        if len(text.strip()) < 2:
            return ""
        
        if len(set(text.replace(' ', ''))) <= 1 and len(text) > 3:
            return ""
        
        roman_patterns = [
            (r'^\s*\(\s*[il1]\s*\)\s*', '(i) '),
            (r'^\s*\(\s*[il1]{2}\s*\)\s*', '(ii) '),
            (r'^\s*\(\s*[il1]{3}\s*\)\s*', '(iii) '),
            (r'^\s*\(\s*[il1]v\s*\)\s*', '(iv) '),
            (r'^\s*\(\s*v\s*\)\s*', '(v) '),
            (r'^\s*\(\s*v[il1]\s*\)\s*', '(vi) '),
            (r'^\s*\(\s*v[il1]{2}\s*\)\s*', '(vii) '),
            (r'^\s*\(\s*v[il1]{3}\s*\)\s*', '(viii) '),
            (r'\b[qQ]\s*[iI]\b', 'Q1'),
            # (r'\bole\b', 'Q2'),
        ]
        
        for pattern, replacement in roman_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                break
        
        if re.match(r'^\s*[Q0OoDG]\s*\d+', text, re.IGNORECASE):
            text = re.sub(r'^\s*[Q0OoDG]\s*(\d+)', r'Q\1.', text, flags=re.IGNORECASE)
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+\(\s*', '/', text)
        
        return text

    def calculate_text_confidence(self, text: str) -> float:
        """Much more lenient confidence calculation"""
        if not text or len(text.strip()) < 2:
            return 0.0
        
        text_clean = text.strip()
        char_count = len(text_clean)
        
        if char_count < 3:
            length_score = 0.6
        elif 3 <= char_count <= 80:
            length_score = 1.0
        elif char_count <= 150:
            length_score = 0.8
        else:
            length_score = 0.5
        
        alpha_chars = sum(1 for c in text_clean if c.isalpha())
        digit_chars = sum(1 for c in text_clean if c.isdigit())
        space_chars = sum(1 for c in text_clean if c.isspace())
        punct_chars = sum(1 for c in text_clean if c in '.,!?;:()-\'\"')
        
        meaningful_chars = alpha_chars + digit_chars + space_chars + punct_chars
        char_ratio = meaningful_chars / char_count if char_count > 0 else 0
        
        special_chars = char_count - meaningful_chars
        if special_chars / char_count > 0.5:
            char_ratio *= 0.7
        
        words = text_clean.split()
        word_score = 0.0
        
        if words:
            valid_words = 0
            total_words = len(words)
            
            for word in words:
                word_clean = word.strip('.,!?;:()-\'\"')
                if len(word_clean) == 0:
                    continue
                
                if word_clean.isdigit():
                    valid_words += 1
                elif len(word_clean) == 1 and word_clean.isalpha():
                    valid_words += 0.9
                elif len(word_clean) >= 2:
                    vowels = sum(1 for c in word_clean.lower() if c in 'aeiou')
                    consonants = sum(1 for c in word_clean.lower() if c.isalpha() and c not in 'aeiou')
                    
                    if vowels > 0:
                        valid_words += 1
                    elif consonants <= 4:
                        valid_words += 0.8
                    else:
                        valid_words += 0.3
            
            word_score = valid_words / total_words if total_words > 0 else 0
        else:
            word_score = 0.5
        
        char_set = set(text_clean.lower().replace(' ', ''))
        min_unique_chars = max(2, len(text_clean) // 6)
        if len(char_set) < min_unique_chars:
            repetition_score = 0.5
        else:
            repetition_score = 1.0
        
        final_confidence = (
            length_score * 0.25 +
            char_ratio * 0.25 +     
            word_score * 0.4 +
            repetition_score * 0.1
        )
        
        return min(1.0, max(0.0, final_confidence))

    def create_text_pdf(self, text_content: str, output_path: str) -> bool:
        """Create a clean PDF with only text content, no page numbers or headers"""
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            styles = getSampleStyleSheet()
            normal_style = styles['Normal']
            normal_style.fontSize = 11
            normal_style.leading = 14
            normal_style.spaceAfter = 6
            
            story = []
            lines = text_content.split('\n')
            
            for line in lines:
                line = line.strip()
                if line:
                    para = Paragraph(line, normal_style)
                    story.append(para)
                    story.append(Spacer(1, 6))
                else:
                    story.append(Spacer(1, 12))
            
            doc.build(story)
            
            print(f"✓ Text PDF created successfully: {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error creating text PDF: {e}")
            import traceback
            traceback.print_exc()
            return False

    def convert_handwritten_file(self, file_bytes: bytes, filename: str) -> Tuple[str, str, float]:
        """Main conversion function that handles both PDF and image files"""
        if not self.trocr_loaded:
            return "", "TrOCR not available", 0.0
        
        try:
            print(f"=== DEBUGGING FILE PROCESSING ===")
            print(f"Received filename: {filename}")
            print(f"File size: {len(file_bytes)} bytes")
            
            # Validate input
            if not filename:
                return "", "No filename provided", 0.0
                
            if len(file_bytes) == 0:
                return "", "File is empty", 0.0
            
            # Check file extension
            file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
            print(f"File extension: {file_ext}")
            
            # Determine file type and get images
            if self.is_pdf_file(filename):
                print(f"✓ Processing PDF file: {filename}")
                page_images = self.pdf_to_images(file_bytes, dpi=350)
            elif self.is_image_file(filename):
                print(f"✓ Processing image file: {filename}")
                print(f"Image file validation result: {self.is_image_file(filename)}")
                # Load single image
                img = self.load_image_from_bytes(file_bytes)
                if img is not None:
                    print(f"✓ Image loaded successfully, shape: {img.shape}")
                    page_images = [img]
                else:
                    print("✗ Failed to load image from bytes")
                    return "", "Failed to load image file", 0.0
            else:
                print(f"✗ Unsupported file format")
                print(f"PDF check result: {self.is_pdf_file(filename)}")
                print(f"Image check result: {self.is_image_file(filename)}")
                return "", f"Unsupported file format: {filename}. Expected PDF, PNG, JPG, or JPEG", 0.0
            
            print(f"Extracted {len(page_images)} page(s)/image(s)")
            
            if not page_images:
                return "", "Failed to extract images from file", 0.0
            
            all_pages_text = []
            total_confidence = 0
            total_lines = 0
            
            for page_num, img in enumerate(page_images):
                print(f"Processing page/image {page_num + 1}...")
                
                if img is None:
                    print(f"Skipping page/image {page_num + 1} - image is None")
                    continue
                
                # Minimal preprocessing
                enhanced_img = self.advanced_image_preprocessing(img)
                if enhanced_img is None:
                    print(f"Skipping page/image {page_num + 1} - preprocessing failed")
                    continue
                
                # Robust text region detection
                text_regions = self.detect_actual_text_regions(enhanced_img)
                print(f"Page/Image {page_num + 1}: Detected {len(text_regions)} text regions")
                
                page_text_lines = []
                line_confidences = []
                
                # Process each region individually
                for i, region in enumerate(text_regions):
                    line_img = self.preprocess_line_for_ocr(enhanced_img, region)
                    
                    if line_img is not None:
                        line_text, line_conf = self.extract_text_single_line(line_img, i+1)
                        
                        if line_text.strip():
                            page_text_lines.append(line_text)
                            line_confidences.append(line_conf)
                            total_lines += 1
                
                # Combine page text
                page_text = '\n'.join(page_text_lines)
                all_pages_text.append(page_text)
                
                # Calculate page confidence
                page_confidence = np.mean(line_confidences) if line_confidences else 0.0
                total_confidence += page_confidence
                
                print(f"Page/Image {page_num + 1}: Extracted {len(page_text_lines)} valid lines")
            
            # Format final text
            final_text = self.format_extracted_text(all_pages_text)
            
            # Calculate overall confidence
            overall_confidence = total_confidence / len(page_images) if page_images else 0.0
            
            print(f"=== FINAL RESULTS ===")
            print(f"Total text length: {len(final_text)}")
            print(f"Overall confidence: {overall_confidence:.3f}")
            print(f"Total lines processed: {total_lines}")
            
            if not final_text.strip():
                return "", "No readable text found in the document", 0.0
            
            return final_text, "", overall_confidence
            
        except Exception as e:
            print(f"✗ Error in convert_handwritten_file: {e}")
            import traceback
            traceback.print_exc()
            return "", f"Error processing file: {str(e)}", 0.0

    def format_extracted_text(self, pages_text: List[str]) -> str:
        """Format extracted text without page numbers"""
        formatted_output = []
        
        for page_num, page_text in enumerate(pages_text, 1):
            if not page_text.strip():
                continue
            
            lines = page_text.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    formatted_output.append(line)
        
        return '\n'.join(formatted_output)

# Initialize converter
print("Initializing HandwritingToTextConverter...")
converter = HandwritingToTextConverter()
print(f"Converter initialized. TrOCR loaded: {converter.trocr_loaded}")

@app.route('/convert-handwritten', methods=['POST'])
def convert_handwritten():
    try:
        print("=== FLASK ROUTE DEBUG START ===")
        
        if 'file' not in request.files:
            print("✗ No 'file' key in request.files")
            print(f"Available keys: {list(request.files.keys())}")
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            print("✗ Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'})
        
        filename = file.filename
        print(f"✓ Received file: '{filename}'")
        print(f"✓ Content type: '{file.content_type}'")
        print(f"✓ Content length: {file.content_length}")
        
        # Check if file is supported format with detailed logging
        filename_lower = filename.lower()
        is_pdf = converter.is_pdf_file(filename)
        is_image = converter.is_image_file(filename)
        
        print(f"File type validation:")
        print(f"  - Original filename: '{filename}'")
        print(f"  - Lowercase filename: '{filename_lower}'")
        print(f"  - PDF check result: {is_pdf}")
        print(f"  - Image check result: {is_image}")
        
        # CRITICAL FIX: Accept both PDFs and images (removing the old restrictive check)
        if not (is_pdf or is_image):
            error_msg = f'File must be a PDF, PNG, JPG, or JPEG. Received: {filename} (content-type: {file.content_type})'
            print(f"✗ File type validation failed: {error_msg}")
            return jsonify({'success': False, 'error': error_msg})
        
        print(f"✓ File type validation passed - File is {'PDF' if is_pdf else 'Image'}")
        
        # Read file bytes with validation
        file_bytes = file.read()
        print(f"✓ File bytes read: {len(file_bytes)} bytes")
        
        # Additional file content validation
        if len(file_bytes) == 0:
            print("✗ File is empty (0 bytes)")
            return jsonify({'success': False, 'error': 'File is empty'})
        
        if len(file_bytes) < 100:  # Very small files are suspicious
            print(f"⚠ Warning: Very small file size: {len(file_bytes)} bytes")
        
        # Validate file signature (magic bytes) for additional security
        file_signature = file_bytes[:8] if len(file_bytes) >= 8 else file_bytes
        print(f"File signature (first 8 bytes): {file_signature}")
        
        # Check for common file signatures
        if is_pdf:
            if not file_bytes.startswith(b'%PDF'):
                print("⚠ Warning: PDF file doesn't start with PDF signature")
        elif is_image:
            png_sig = file_bytes.startswith(b'\x89PNG\r\n\x1a\n')
            jpg_sig = file_bytes.startswith(b'\xff\xd8\xff')
            jpg_sig_alt = file_bytes.startswith(b'\xff\xd8')  # More lenient JPEG check
            if not (png_sig or jpg_sig or jpg_sig_alt):
                print("⚠ Warning: Image file doesn't have expected signature")
                print(f"   - PNG signature check: {png_sig}")
                print(f"   - JPG signature check: {jpg_sig}")
                print(f"   - JPG alt signature check: {jpg_sig_alt}")
                # Don't fail here - proceed with processing anyway
            else:
                print(f"✓ Image signature validated ({'PNG' if png_sig else 'JPEG'})")
        
        # Convert handwritten file to text
        print("=== Starting conversion process ===")
        extracted_text, error_msg, confidence = converter.convert_handwritten_file(file_bytes, filename)
        
        print(f"=== Conversion completed ===")
        print(f"  - Success: {not bool(error_msg)}")
        print(f"  - Error: {error_msg if error_msg else 'None'}")
        print(f"  - Text length: {len(extracted_text) if extracted_text else 0}")
        print(f"  - Confidence: {confidence}")
        
        if error_msg:
            print(f"✗ Conversion failed: {error_msg}")
            return jsonify({'success': False, 'error': error_msg})
        
        if not extracted_text or not extracted_text.strip():
            print("✗ No text extracted from file")
            return jsonify({'success': False, 'error': 'No text could be extracted from the handwritten file'})
        
        print(f"✓ Text extraction successful: {len(extracted_text)} characters")
        print(f"First 200 characters of extracted text: {extracted_text[:200]}...")
        
        # Create temporary PDF file with extracted text
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            temp_pdf_path = tmp_file.name
        
        print(f"Creating text PDF at: {temp_pdf_path}")
        
        # Create text-only PDF
        if converter.create_text_pdf(extracted_text, temp_pdf_path):
            print(f"✓ Text PDF created successfully: {temp_pdf_path}")
            
            # Verify the created PDF
            try:
                import os
                pdf_size = os.path.getsize(temp_pdf_path)
                print(f"✓ Created PDF size: {pdf_size} bytes")
                
                if pdf_size < 100:
                    print("⚠ Warning: Created PDF is very small")
                else:
                    # Verify it's a valid PDF by reading first few bytes
                    with open(temp_pdf_path, 'rb') as pdf_file:
                        pdf_header = pdf_file.read(4)
                        if pdf_header.startswith(b'%PDF'):
                            print("✓ Created PDF has valid signature")
                        else:
                            print("⚠ Warning: Created PDF may not have valid signature")
                            
            except Exception as size_error:
                print(f"Could not verify PDF size: {size_error}")
            
            # Return the PDF file
            return send_file(
                temp_pdf_path,
                as_attachment=True,
                download_name=f'converted_{filename.split(".")[0]}.pdf',
                mimetype='application/pdf'
            )
        else:
            print("✗ Failed to create text PDF")
            return jsonify({'success': False, 'error': 'Failed to create text PDF'})
            
    except Exception as e:
        print(f"✗ Unexpected error in convert_handwritten: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})
    finally:
        print("=== FLASK ROUTE DEBUG END ===\n")

@app.route('/health', methods=['GET'])
def health_check():
    health_status = {
        'status': 'healthy',
        'trocr_available': converter.trocr_loaded
    }
    print(f"Health check requested: {health_status}")
    return jsonify(health_status)

if __name__ == '_main_':
    print("=== Starting Handwriting to Text Conversion Service ===")
    print(f"TrOCR Available: {converter.trocr_loaded}")
    print(f"Device: {converter.device if converter.trocr_loaded else 'N/A'}")
    if converter.trocr_loaded:
        print("✓ Service ready for handwritten text conversion")
    else:
        print("✗ Service started but TrOCR is not available")
    print("Server starting on 0.0.0.0:5001...")
    app.run(debug=False, host='0.0.0.0', port=5001)