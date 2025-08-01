from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import base64
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'}

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class OCRProcessor:
    """
    OCR Processing class - This will be replaced with your actual OCR system
    """
    def __init__(self):
        # Initialize your OCR model/system here
        # Example: self.model = load_your_ocr_model()
        pass
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for better OCR results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply adaptive thresholding
            processed = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Optional: Apply morphological operations
            kernel = np.ones((1,1), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            
            return processed
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None
    
    def extract_text(self, image_path):
        """
        Extract text from image - Replace this with your OCR system
        """
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_path)
            
            if processed_image is None:
                return None, 0.0, None
            
            # Save processed image
            processed_filename = f"processed_{uuid.uuid4().hex}.png"
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            cv2.imwrite(processed_path, processed_image)
            
            # TODO: Replace this section with your actual OCR system
            # Example integration points:
            
            # For TrOCR:
            # text, confidence = your_trocr_model.predict(processed_image)
            
            # For EasyOCR:
            # results = your_easyocr_reader.readtext(processed_image)
            # text = ' '.join([result[1] for result in results])
            # confidence = np.mean([result[2] for result in results])
            
            # For Tesseract:
            # import pytesseract
            # text = pytesseract.image_to_string(processed_image)
            # confidence = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
            
            # Mock implementation for now
            mock_text = "This is a placeholder text. Replace with your OCR system output."
            mock_confidence = 0.85
            
            return mock_text, mock_confidence, processed_path
            
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return None, 0.0, None

# Initialize OCR processor
ocr_processor = OCRProcessor()

@app.route('/')
def index():
    """Serve the main UI"""
    return send_from_directory('.', 'index.html')

@app.route('/api/process-image', methods=['POST'])
def process_image():
    """
    Main endpoint for processing uploaded images
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use PNG, JPG, JPEG, TIFF, BMP, or GIF'}), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save uploaded file
        file.save(filepath)
        logger.info(f"File saved: {filepath}")
        
        # Process image with OCR
        recognized_text, confidence, processed_image_path = ocr_processor.extract_text(filepath)
        
        if recognized_text is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        # Convert processed image to base64 for frontend display
        processed_image_b64 = None
        if processed_image_path and os.path.exists(processed_image_path):
            with open(processed_image_path, 'rb') as img_file:
                processed_image_b64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Prepare response
        response_data = {
            'success': True,
            'recognizedText': recognized_text,
            'confidence': float(confidence),
            'processedImage': f"data:image/png;base64,{processed_image_b64}" if processed_image_b64 else None,
            'fileName': filename
        }
        
        # Clean up temporary files (optional)
        # os.remove(filepath)
        # if processed_image_path:
        #     os.remove(processed_image_path)
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'OCR API is running',
        'version': '1.0.0'
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)