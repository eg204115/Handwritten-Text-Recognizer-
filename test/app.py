from flask import Flask, render_template_string, request, jsonify, send_file
import os
import cv2
import numpy as np
import base64
import io
from PIL import Image, ImageEnhance
import pytesseract
import warnings
import tempfile
warnings.filterwarnings('ignore')

# TrOCR imports
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False

# Import HandwritingToTextConverter
try:
    from pdf_converter import HandwritingToTextConverter
    HANDWRITING_CONVERTER_AVAILABLE = True
except ImportError:
    HANDWRITING_CONVERTER_AVAILABLE = False
    print("Warning: HandwritingToTextConverter not available")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class OCRProcessor:
    def __init__(self):
        self.trocr_loaded = False
        if TROCR_AVAILABLE:
            try:
                print("Loading TrOCR models...")
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.processor_hw = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
                self.model_hw = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(self.device)
                self.processor_pr = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
                self.model_pr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').to(self.device)
                self.trocr_loaded = True
                print("TrOCR models loaded successfully!")
            except Exception as e:
                print(f"Error loading TrOCR: {e}")
                self.trocr_loaded = False

        self.tesseract_configs = {
            'handwriting_psm6': '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;:()-\'\"',
            'handwriting_psm7': '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;:()-\'\"',
            'lstm_only': '--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;:()-\'\"\n'
        }

    def detect_text_lines(self, img):
        """Detect text lines using horizontal projection"""
        print('Detect text lines using horizontal projection')
        if img is None:
            return []
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        if np.mean(gray) > 127:
            binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        else:
            binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            binary = cv2.bitwise_not(binary)
        
        horizontal_projection = np.sum(binary, axis=1)
        
        lines = []
        in_line = False
        line_start = 0
        min_line_height = max(8, img.shape[0] // 50)
        
        for i, projection in enumerate(horizontal_projection):
            if projection > 0 and not in_line:
                line_start = i
                in_line = True
            elif projection == 0 and in_line:
                if i - line_start >= min_line_height:
                    lines.append((line_start, i))
                in_line = False
        
        if in_line and img.shape[0] - line_start >= min_line_height:
            lines.append((line_start, img.shape[0]))
        
        if not lines:
            return [(0, img.shape[0])]
        
        merged_lines = []
        for start, end in lines:
            if merged_lines and start - merged_lines[-1][1] < min_line_height // 2:
                merged_lines[-1] = (merged_lines[-1][0], end)
            else:
                merged_lines.append((start, end))
        
        return merged_lines

    def extract_line_image(self, img, line_coords, padding=5):
        """Extract individual line image with padding"""
        print('Extract individual line image with padding')
        start_y, end_y = line_coords
        start_y = max(0, start_y - padding)
        end_y = min(img.shape[0], end_y + padding)
        
        line_img = img[start_y:end_y, :]
        
        if line_img.shape[0] < 20:
            pad_needed = (20 - line_img.shape[0]) // 2
            line_img = cv2.copyMakeBorder(line_img, pad_needed, pad_needed, 0, 0, 
                                        cv2.BORDER_CONSTANT, value=255)
        
        return line_img

    def correct_skew_advanced(self, img):
        """Advanced skew correction"""
        print('Advanced skew correction')
        try:
            edges1 = cv2.Canny(img, 30, 100, apertureSize=3)
            edges2 = cv2.Canny(img, 50, 150, apertureSize=3)
            edges = cv2.bitwise_or(edges1, edges2)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(x2 - x1) > 10:
                        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                        if -15 < angle < 15:
                            angles.append(angle)
                
                if len(angles) > 5:
                    median_angle = np.median(angles)
                    if abs(median_angle) > 0.3:
                        (h, w) = img.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return img
        except:
            return img

    def enhanced_preprocessing_multiline(self, img):
        """Multi-line preprocessing"""
        print('Multi-line preprocessing')
        if img is None:
            return None, []

        orig_h, orig_w = img.shape[:2]

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Multi-stage preprocessing
        img = cv2.medianBlur(img, 3)
        img = cv2.bilateralFilter(img, 9, 80, 80)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # Thresholding
        _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)
        combined = cv2.bitwise_and(otsu, adaptive_mean)
        
        # Morphological operations
        kernel_conn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_conn)
        
        # Component filtering
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)
        min_area = max(12, (orig_h * orig_w) // 6000)
        
        mask = np.zeros(combined.shape, dtype=np.uint8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            width = stats[i, cv2.CC_STAT_WIDTH]
            aspect_ratio = width / height if height > 0 else 0
            
            if (area >= min_area or (area >= 8 and height >= 4 and aspect_ratio < 5) or 
                (area >= 4 and height <= 8 and width <= 8)):
                mask[labels == i] = 255

        # Skew correction
        mask = self.correct_skew_advanced(mask)
        
        # Detect lines
        line_coords = self.detect_text_lines(mask)
        
        # Extract line images
        line_images = []
        for coords in line_coords:
            line_img = self.extract_line_image(mask, coords)
            
            # Resize for TrOCR
            height, width = line_img.shape
            target_height = 160
            scale_factor = target_height / height
            new_width = int(width * scale_factor)
            
            line_img = cv2.resize(line_img, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
            line_img = cv2.copyMakeBorder(line_img, 30, 30, 50, 50, cv2.BORDER_CONSTANT, value=255)
            
            if np.mean(line_img) < 127:
                line_img = cv2.bitwise_not(line_img)
                
            line_images.append(line_img)
        
        return mask, line_images


    def post_process_text(self, text):
        """Post-process OCR text"""
        print('Post-process OCR text')
        if not text:
            return ""
        
        text = text.strip()
        text = ' '.join(text.split())
        text = text.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?')
        
        # Basic character fixes
        char_fixes = {'rn': 'm', 'cl': 'd', 'li': 'h', 'vv': 'w', '0': 'o', '1': 'l', '5': 's'}
        for wrong, right in char_fixes.items():
            text = text.replace(wrong, right)
        
        return text.strip()

    def extract_text_with_trocr(self, line_images, model_type='handwritten'):
        """Extract text using TrOCR"""
        print('Extract text using TrOCR')
        if not line_images or not self.trocr_loaded:
            return "", 0.0
        
        processor = self.processor_hw if model_type == 'handwritten' else self.processor_pr
        model = self.model_hw if model_type == 'handwritten' else self.model_pr
        
        line_texts = []
        confidences = []
        
        for line_img in line_images:
            try:
                pil_img = Image.fromarray(line_img).convert('RGB')
                
                # Enhanced strategies
                enhancer = ImageEnhance.Contrast(pil_img)
                enhanced1 = enhancer.enhance(1.3)
                enhancer = ImageEnhance.Sharpness(enhanced1)
                enhanced1 = enhancer.enhance(1.5)
                
                pixel_values = processor(enhanced1, return_tensors="pt").pixel_values.to(self.device)
                with torch.no_grad():
                    generated_ids = model.generate(
                        pixel_values, max_length=80, num_beams=4,
                        early_stopping=True, do_sample=False,
                        length_penalty=0.9, repetition_penalty=1.1
                    )
                
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                text = self.post_process_text(text)
                
                if text.strip():
                    confidence = self.calculate_confidence(text, enhanced1)
                    line_texts.append(text)
                    confidences.append(confidence)
                    
            except Exception as e:
                continue
        
        final_text = '\n'.join(line_texts) if line_texts else ""
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return final_text, avg_confidence



    def extract_text_with_tesseract(self, img, line_images=None):
        """Extract text using Tesseract - Modified to handle multi-line"""
        print('Extract text using Tesseract - Modified to handle multi-line')
        if img is None:
            return "", 0.0
        
        # If line_images are provided, process each line separately
        if line_images and len(line_images) > 0:
            line_texts = []
            confidences = []
            
            for line_img in line_images:
                pil_img = Image.fromarray(line_img)
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(1.3)
                enhancer = ImageEnhance.Sharpness(pil_img)
                pil_img = enhancer.enhance(1.4)
                
                best_text = ""
                best_confidence = 0.0
                
                for config_name, config in self.tesseract_configs.items():
                    try:
                        text = pytesseract.image_to_string(pil_img, config=config)
                        text = self.post_process_text(text)
                        if len(text.strip()) > 1:  # More lenient minimum
                            confidence = self.calculate_confidence(text)
                            if confidence > best_confidence:
                                best_text = text
                                best_confidence = confidence
                    except:
                        continue
                
                if best_text.strip():
                    line_texts.append(best_text.strip())
                    confidences.append(best_confidence)
            
            final_text = '\n'.join(line_texts) if line_texts else ""
            avg_confidence = np.mean(confidences) if confidences else 0.0
            return final_text, avg_confidence
        
        # Fallback to original single image processing
        pil_img = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.3)
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.4)
        
        results = []
        for config_name, config in self.tesseract_configs.items():
            try:
                text = pytesseract.image_to_string(pil_img, config=config)
                text = self.post_process_text(text)
                if len(text.strip()) > 2:
                    confidence = self.calculate_confidence(text)
                    results.append({'text': text, 'confidence': confidence})
            except:
                continue
        
        if results:
            best_result = max(results, key=lambda x: x['confidence'])
            return best_result['text'], best_result['confidence']
        
        return "", 0.0

    def process_image(self, img, method='hybrid'):
        """Process image with specified method - TrOCR-first hybrid with intelligent fallback"""
        print('Process image with specified method - TrOCR-first hybrid with intelligent fallback')
        processed_img, line_images = self.enhanced_preprocessing_multiline(img)
        
        if method == 'trocr':
            if self.trocr_loaded and line_images:
                text, confidence = self.extract_text_with_trocr(line_images)
            else:
                return "TrOCR not available", processed_img, 0.0
                
        elif method == 'tesseract':
            # Pass line_images to Tesseract for multi-line processing
            text, confidence = self.extract_text_with_tesseract(processed_img, line_images)
            
        else:  # hybrid - TrOCR-first with intelligent fallback
            if not self.trocr_loaded:
                # No TrOCR available, use Tesseract
                text, confidence = self.extract_text_with_tesseract(processed_img, line_images)
                return text, processed_img, confidence
            
            # Step 1: Try TrOCR first (it's generally better for handwriting)
            trocr_text, trocr_conf = self.extract_text_with_trocr(line_images)
            
            # Step 2: Check if TrOCR result is acceptable
            trocr_acceptable = self.is_result_acceptable(trocr_text, trocr_conf)
            
            if trocr_acceptable:
                # TrOCR result is good, use it
                text, confidence = trocr_text, trocr_conf
            else:
                # TrOCR struggled, try Tesseract as fallback
                tesseract_text, tesseract_conf = self.extract_text_with_tesseract(processed_img, line_images)
                
                # Step 3: Compare fallback with original
                if tesseract_conf > trocr_conf * 1.3 and len(tesseract_text.strip()) > len(trocr_text.strip()) * 0.5:
                    # Tesseract significantly better and has reasonable content
                    text, confidence = tesseract_text, tesseract_conf
                elif len(tesseract_text.strip()) > len(trocr_text.strip()) * 2 and tesseract_conf > 0.2:
                    # Tesseract found much more text with decent confidence
                    text, confidence = tesseract_text, tesseract_conf
                else:
                    # Keep TrOCR result even if confidence is low (it might still be better)
                    text, confidence = trocr_text, trocr_conf
        
        return text, processed_img, confidence

    def is_result_acceptable(self, text, confidence):
        """Check if OCR result is acceptable or needs fallback"""
        print('Check if OCR result is acceptable or needs fallback')
        if not text or not text.strip():
            return False
        
        text_clean = text.strip()
        
        # Very low confidence - definitely not acceptable
        if confidence < 0.15:
            return False
        
        # Very short text with low confidence
        if len(text_clean) < 3 and confidence < 0.4:
            return False
        
        # Check for garbled output (too many non-alphanumeric characters)
        alphanumeric = sum(1 for c in text_clean if c.isalnum() or c.isspace())
        total_chars = len(text_clean)
        
        if total_chars > 0:
            meaningful_ratio = alphanumeric / total_chars
            if meaningful_ratio < 0.6:  # Less than 60% meaningful characters
                return False
        
        # Check for repetitive patterns (sign of poor OCR)
        words = text_clean.split()
        if len(words) > 3:
            unique_words = len(set(words))
            if unique_words / len(words) < 0.5:  # Too many repeated words
                return False
        
        # If we reach here, the result seems acceptable
        return True

    def calculate_confidence(self, text, image=None):
        """Calculate text confidence - Enhanced for better TrOCR vs Tesseract comparison"""
        print('Calculate text confidence - Enhanced for better TrOCR vs Tesseract comparison')
        if not text:
            return 0.0
        
        text_clean = text.strip()
        if not text_clean:
            return 0.0
            
        char_count = len(text_clean)
        
        # Length score - more generous for reasonable lengths
        if char_count < 2:
            length_score = 0.1
        elif char_count <= 50:
            length_score = 1.0
        elif char_count <= 150:
            length_score = 0.9
        else:
            length_score = max(0.4, 1.0 - (char_count - 150) / 300)
        
        # Character type analysis
        alpha_chars = sum(1 for c in text_clean if c.isalpha())
        digit_chars = sum(1 for c in text_clean if c.isdigit())
        space_chars = sum(1 for c in text_clean if c.isspace())
        punct_chars = sum(1 for c in text_clean if c in '.,!?;:()-\'\"')
        
        meaningful_chars = alpha_chars + digit_chars + space_chars + punct_chars
        char_ratio = meaningful_chars / char_count if char_count > 0 else 0
        
        # Word structure analysis
        words = [w.strip('.,!?;:()-\'\"') for w in text_clean.split() if w.strip()]
        word_score = 0.0
        
        if words:
            valid_words = 0
            for word in words:
                if len(word) >= 1:
                    if word.isdigit():  # Numbers are always valid
                        valid_words += 1
                    elif len(word) == 1 and word.isalpha():  # Single letters OK
                        valid_words += 1
                    elif len(word) >= 2:
                        # Check for reasonable letter distribution
                        vowels = sum(1 for c in word.lower() if c in 'aeiou')
                        consonants = sum(1 for c in word.lower() if c.isalpha() and c not in 'aeiou')
                        
                        if vowels > 0:  # Has vowels - likely real word
                            valid_words += 1
                        elif len(word) <= 4 and consonants >= 2:  # Short consonant clusters OK
                            valid_words += 1
                        elif consonants >= 1:  # At least some structure
                            valid_words += 0.5
            
            word_score = min(1.0, valid_words / len(words))
        
        # Multi-line bonus (encourages proper line detection)
        lines = [line.strip() for line in text_clean.split('\n') if line.strip()]
        line_bonus = min(0.1, (len(lines) - 1) * 0.03) if len(lines) > 1 else 0
        
        # Penalize very repetitive content
        repetition_penalty = 0
        if len(words) > 2:
            unique_words = len(set(word.lower() for word in words))
            if unique_words / len(words) < 0.7:
                repetition_penalty = 0.1
        
        # Final confidence calculation
        final_confidence = (
            length_score * 0.25 + 
            char_ratio * 0.35 + 
            word_score * 0.35 + 
            line_bonus - 
            repetition_penalty
        )
        
        return min(1.0, max(0.0, final_confidence))

# Initialize OCR processor
ocr_processor = OCRProcessor()

# Initialize Handwriting Converter
if HANDWRITING_CONVERTER_AVAILABLE:
    print("Initializing HandwritingToTextConverter...")
    handwriting_converter = HandwritingToTextConverter()
    print(f"Handwriting Converter initialized. TrOCR loaded: {handwriting_converter.trocr_loaded}")
else:
    handwriting_converter = None
    print("Warning: HandwritingToTextConverter not available")

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})
        
        method = request.form.get('method', 'hybrid')
        
        # Read image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'error': 'Invalid image format'})
        
        # Process image
        text, processed_img, confidence = ocr_processor.process_image(img, method)
        
        # Convert processed image to base64
        _, buffer = cv2.imencode('.png', processed_img)
        processed_img_b64 = base64.b64encode(buffer).decode('utf-8')
        print('extracted text', text)
        
        return jsonify({
            'success': True,
            'text': text,
            'confidence': confidence,
            'processed_image': processed_img_b64
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/convert-handwritten', methods=['POST'])
def convert_handwritten():
    try:
        print("=== HANDWRITTEN CONVERSION DEBUG START ===")
        
        if handwriting_converter is None:
            return jsonify({'success': False, 'error': 'Handwriting converter not available'})
        
        if 'file' not in request.files:
            print("✗ No 'file' key in request.files")
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            print("✗ Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'})
        
        filename = file.filename
        print(f"✓ Received file: '{filename}'")
        
        # Read file bytes
        file_bytes = file.read()
        print(f"✓ File bytes read: {len(file_bytes)} bytes")
        
        if len(file_bytes) == 0:
            return jsonify({'success': False, 'error': 'File is empty'})
        
        # Convert handwritten file to text
        extracted_text, error_msg, confidence = handwriting_converter.convert_handwritten_file(file_bytes, filename)
        
        if error_msg:
            return jsonify({'success': False, 'error': error_msg})
        
        if not extracted_text or not extracted_text.strip():
            return jsonify({'success': False, 'error': 'No text could be extracted from the file'})
        
        print(f"✓ Text extraction successful: {len(extracted_text)} characters")
        print(f"✓ Confidence: {confidence:.3f}")
        print(f"✓ First 200 chars: {extracted_text[:200]}...")
        
        # Return only the extracted text as JSON
        return jsonify({
            'success': True,
            'text': extracted_text,
            'confidence': confidence,
            'filename': filename,
            'message': 'PDF processed successfully - text extracted'
        })
            
    except Exception as e:
        print(f"✗ Unexpected error in convert_handwritten: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})
    finally:
        print("=== HANDWRITTEN CONVERSION DEBUG END ===\n")
        

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'trocr_available': ocr_processor.trocr_loaded,
        'handwriting_converter_available': handwriting_converter.trocr_loaded if handwriting_converter else False
    })

if __name__ == '__main__':
    print("Starting Enhanced Multi-Line OCR System...")
    print(f"TrOCR Available: {ocr_processor.trocr_loaded}")
    print(f"Handwriting Converter Available: {handwriting_converter.trocr_loaded if handwriting_converter else False}")
    app.run(debug=True, host='0.0.0.0', port=5000)