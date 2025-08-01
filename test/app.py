# # from flask import Flask, request, jsonify, render_template_string
# # from flask_cors import CORS
# # import base64
# # import io
# # import os
# # import cv2
# # import numpy as np
# # from PIL import Image, ImageEnhance
# # import tempfile
# # import logging

# # # Set up logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # # Check for optional dependencies
# # TROCR_AVAILABLE = False
# # try:
# #     import torch
# #     from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# #     TROCR_AVAILABLE = True
# #     logger.info("TrOCR dependencies found")
# # except ImportError as e:
# #     logger.warning(f"TrOCR not available: {e}")

# # TESSERACT_AVAILABLE = False
# # try:
# #     import pytesseract
# #     TESSERACT_AVAILABLE = True
# #     logger.info("Tesseract available")
# # except ImportError as e:
# #     logger.warning(f"Tesseract not available: {e}")

# # from collections import Counter

# # app = Flask(__name__)
# # CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])

# # # HTML Template for the interface
# # HTML_TEMPLATE = """
# # <!DOCTYPE html>
# # <html lang="en">
# # <head>
# #     <meta charset="UTF-8">
# #     <meta name="viewport" content="width=device-width, initial-scale=1.0">
# #     <title>Enhanced OCR API</title>
# #     <style>
# #         * {
# #             margin: 0;
# #             padding: 0;
# #             box-sizing: border-box;
# #         }
        
# #         body {
# #             font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
# #             line-height: 1.6;
# #             color: #333;
# #             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
# #             min-height: 100vh;
# #             padding: 20px;
# #         }
        
# #         .container {
# #             max-width: 800px;
# #             margin: 0 auto;
# #             background: white;
# #             border-radius: 20px;
# #             box-shadow: 0 20px 40px rgba(0,0,0,0.1);
# #             overflow: hidden;
# #         }
        
# #         .header {
# #             background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
# #             color: white;
# #             padding: 30px;
# #             text-align: center;
# #         }
        
# #         .header h1 {
# #             font-size: 2.5em;
# #             margin-bottom: 10px;
# #             font-weight: 300;
# #         }
        
# #         .status-bar {
# #             display: flex;
# #             align-items: center;
# #             justify-content: center;
# #             gap: 10px;
# #             margin-top: 15px;
# #         }
        
# #         .status-icon {
# #             width: 12px;
# #             height: 12px;
# #             border-radius: 50%;
# #             background: #4CAF50;
# #         }
        
# #         .status-icon.loading {
# #             background: #ff9800;
# #             animation: pulse 1.5s infinite;
# #         }
        
# #         .status-icon.error {
# #             background: #f44336;
# #         }
        
# #         @keyframes pulse {
# #             0% { opacity: 1; }
# #             50% { opacity: 0.5; }
# #             100% { opacity: 1; }
# #         }
        
# #         .main-content {
# #             padding: 40px;
# #         }
        
# #         .upload-section {
# #             margin-bottom: 30px;
# #         }
        
# #         .upload-area {
# #             border: 3px dashed #ddd;
# #             border-radius: 15px;
# #             padding: 40px;
# #             text-align: center;
# #             cursor: pointer;
# #             transition: all 0.3s ease;
# #             background: #fafafa;
# #         }
        
# #         .upload-area:hover, .upload-area.dragover {
# #             border-color: #4facfe;
# #             background: #f0f9ff;
# #             transform: translateY(-2px);
# #         }
        
# #         .upload-icon {
# #             font-size: 3em;
# #             margin-bottom: 15px;
# #         }
        
# #         .upload-text {
# #             font-size: 1.1em;
# #             margin-bottom: 10px;
# #         }
        
# #         .method-selection {
# #             margin-bottom: 30px;
# #             background: #f8f9fa;
# #             padding: 20px;
# #             border-radius: 10px;
# #         }
        
# #         .method-options {
# #             display: flex;
# #             gap: 20px;
# #             justify-content: center;
# #             flex-wrap: wrap;
# #         }
        
# #         .method-option {
# #             display: flex;
# #             align-items: center;
# #             gap: 8px;
# #             padding: 10px 15px;
# #             background: white;
# #             border: 2px solid #e0e0e0;
# #             border-radius: 8px;
# #             cursor: pointer;
# #             transition: all 0.3s ease;
# #         }
        
# #         .method-option:hover {
# #             border-color: #4facfe;
# #         }
        
# #         .method-option input[type="radio"]:checked + label {
# #             border-color: #4facfe;
# #             background: #f0f9ff;
# #         }
        
# #         .preview-container {
# #             display: none;
# #             margin: 20px 0;
# #             text-align: center;
# #         }
        
# #         .preview-image {
# #             max-width: 100%;
# #             max-height: 300px;
# #             border-radius: 10px;
# #             box-shadow: 0 5px 15px rgba(0,0,0,0.1);
# #         }
        
# #         .process-btn {
# #             background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
# #             color: white;
# #             border: none;
# #             padding: 15px 40px;
# #             font-size: 1.1em;
# #             border-radius: 50px;
# #             cursor: pointer;
# #             transition: all 0.3s ease;
# #             display: block;
# #             margin: 20px auto;
# #             min-width: 200px;
# #         }
        
# #         .process-btn:hover:not(:disabled) {
# #             transform: translateY(-2px);
# #             box-shadow: 0 10px 25px rgba(79, 172, 254, 0.3);
# #         }
        
# #         .process-btn:disabled {
# #             opacity: 0.6;
# #             cursor: not-allowed;
# #         }
        
# #         .loading {
# #             display: none;
# #             text-align: center;
# #             padding: 20px;
# #         }
        
# #         .loading-spinner {
# #             width: 40px;
# #             height: 40px;
# #             border: 4px solid #f3f3f3;
# #             border-top: 4px solid #4facfe;
# #             border-radius: 50%;
# #             animation: spin 1s linear infinite;
# #             margin: 0 auto 15px;
# #         }
        
# #         @keyframes spin {
# #             0% { transform: rotate(0deg); }
# #             100% { transform: rotate(360deg); }
# #         }
        
# #         .error-message {
# #             display: none;
# #             background: #ffebee;
# #             color: #c62828;
# #             padding: 15px;
# #             border-radius: 8px;
# #             border-left: 4px solid #f44336;
# #             margin: 20px 0;
# #         }
        
# #         .results-section {
# #             display: none;
# #             margin-top: 30px;
# #             padding: 25px;
# #             background: #f8f9fa;
# #             border-radius: 15px;
# #             border: 1px solid #e9ecef;
# #         }
        
# #         .result-header {
# #             display: flex;
# #             justify-content: space-between;
# #             align-items: center;
# #             margin-bottom: 15px;
# #         }
        
# #         .confidence-badge {
# #             padding: 5px 15px;
# #             border-radius: 20px;
# #             font-size: 0.9em;
# #             font-weight: bold;
# #             background: #4CAF50;
# #             color: white;
# #         }
        
# #         .confidence-badge.confidence-low {
# #             background: #f44336;
# #         }
        
# #         .confidence-badge.confidence-medium {
# #             background: #ff9800;
# #         }
        
# #         .result-text {
# #             background: white;
# #             padding: 20px;
# #             border-radius: 10px;
# #             border: 1px solid #ddd;
# #             font-size: 1.1em;
# #             line-height: 1.8;
# #             min-height: 100px;
# #             word-wrap: break-word;
# #         }
        
# #         .copy-btn {
# #             background: #4CAF50;
# #             color: white;
# #             border: none;
# #             padding: 10px 20px;
# #             border-radius: 5px;
# #             cursor: pointer;
# #             margin-top: 10px;
# #             transition: all 0.3s ease;
# #         }
        
# #         .copy-btn:hover {
# #             background: #45a049;
# #         }
        
# #         .hidden {
# #             display: none;
# #         }
        
# #         #file-input {
# #             display: none;
# #         }
        
# #         @media (max-width: 600px) {
# #             .container {
# #                 margin: 10px;
# #                 border-radius: 15px;
# #             }
            
# #             .main-content {
# #                 padding: 20px;
# #             }
            
# #             .method-options {
# #                 flex-direction: column;
# #                 align-items: center;
# #             }
# #         }
# #     </style>
# # </head>
# # <body>
# #     <div class="container">
# #         <div class="header">
# #             <h1>🔍 Enhanced OCR API</h1>
# #             <p>Advanced Handwritten & Printed Text Recognition</p>
# #             <div class="status-bar">
# #                 <div id="api-status" class="status-icon loading"></div>
# #                 <span id="api-status-text">Checking API...</span>
# #                 <span id="method-info" style="margin-left: 10px; opacity: 0.8;"></span>
# #             </div>
# #         </div>
        
# #         <div class="main-content">
# #             <!-- Upload Section -->
# #             <div class="upload-section">
# #                 <div id="upload-area" class="upload-area">
# #                     <div class="upload-icon">📁</div>
# #                     <div class="upload-text">
# #                         <strong>Click to upload</strong> or drag and drop your image here
# #                     </div>
# #                     <div style="color: #888; font-size: 0.9em;">
# #                         Supports JPG, PNG, GIF, BMP (max 5MB)
# #                     </div>
# #                 </div>
# #                 <input type="file" id="file-input" accept="image/*">
# #             </div>
            
# #             <!-- Preview -->
# #             <div id="preview-container" class="preview-container">
# #                 <img id="preview-image" class="preview-image" alt="Preview">
# #             </div>
            
# #             <!-- Method Selection -->
# #             <div class="method-selection">
# #                 <h3 style="margin-bottom: 15px; text-align: center;">Choose OCR Method</h3>
# #                 <div class="method-options">
# #                     <label class="method-option">
# #                         <input type="radio" name="method" value="hybrid" checked>
# #                         <span>🔧 Hybrid (Recommended)</span>
# #                     </label>
# #                     <label class="method-option" for="trocr">
# #                         <input type="radio" name="method" value="trocr" id="trocr">
# #                         <span>🤖 TrOCR (Handwritten)</span>
# #                     </label>
# #                     <label class="method-option">
# #                         <input type="radio" name="method" value="tesseract">
# #                         <span>📝 Tesseract (Traditional)</span>
# #                     </label>
# #                 </div>
# #             </div>
            
# #             <!-- Process Button -->
# #             <button id="process-btn" class="process-btn" disabled onclick="processImage()">
# #                 🚀 Process Image
# #             </button>
            
# #             <!-- Loading -->
# #             <div id="loading" class="loading">
# #                 <div class="loading-spinner"></div>
# #                 <p>Processing your image...</p>
# #             </div>
            
# #             <!-- Error Messages -->
# #             <div id="error-message" class="error-message"></div>
            
# #             <!-- Results -->
# #             <div id="results-section" class="results-section">
# #                 <div class="result-header">
# #                     <h3>📋 Extracted Text</h3>
# #                     <span id="confidence-badge" class="confidence-badge">0% Confidence</span>
# #                 </div>
# #                 <div id="result-text" class="result-text"></div>
# #                 <button class="copy-btn" onclick="copyToClipboard()">📋 Copy Text</button>
# #             </div>
# #         </div>
# #     </div>

# #     <script>
# #         let selectedFile = null;
# #         const API_BASE = window.location.origin;

# #         document.addEventListener('DOMContentLoaded', function() {
# #             checkAPIStatus();
# #         });

# #         async function checkAPIStatus() {
# #             const statusIcon = document.getElementById('api-status');
# #             const statusText = document.getElementById('api-status-text');
            
# #             try {
# #                 statusIcon.className = 'status-icon loading';
# #                 statusText.textContent = 'Checking API...';

# #                 const response = await fetch(`${API_BASE}/health`);
# #                 const data = await response.json();

# #                 if (data.status === 'healthy') {
# #                     statusIcon.className = 'status-icon';
# #                     statusText.textContent = `API Ready (${data.device || 'CPU'})`;
                    
# #                     const methodInfo = document.getElementById('method-info');
# #                     if (data.trocr_available) {
# #                         methodInfo.textContent = 'TrOCR + Tesseract Available';
# #                     } else {
# #                         methodInfo.textContent = 'Tesseract Only';
# #                         document.getElementById('trocr').disabled = true;
# #                         document.querySelector('label[for="trocr"]').style.opacity = '0.5';
# #                     }
# #                 } else {
# #                     throw new Error('API not healthy');
# #                 }
# #             } catch (error) {
# #                 statusIcon.className = 'status-icon error';
# #                 statusText.textContent = 'API Unavailable';
# #                 showError('Could not connect to OCR API. Please check if the server is running.');
# #             }
# #         }

# #         const uploadArea = document.getElementById('upload-area');
# #         const fileInput = document.getElementById('file-input');
# #         const previewContainer = document.getElementById('preview-container');
# #         const previewImage = document.getElementById('preview-image');
# #         const processBtn = document.getElementById('process-btn');

# #         uploadArea.addEventListener('click', () => fileInput.click());

# #         uploadArea.addEventListener('dragover', (e) => {
# #             e.preventDefault();
# #             uploadArea.classList.add('dragover');
# #         });

# #         uploadArea.addEventListener('dragleave', () => {
# #             uploadArea.classList.remove('dragover');
# #         });

# #         uploadArea.addEventListener('drop', (e) => {
# #             e.preventDefault();
# #             uploadArea.classList.remove('dragover');
# #             const files = e.dataTransfer.files;
# #             if (files.length > 0) {
# #                 handleFile(files[0]);
# #             }
# #         });

# #         fileInput.addEventListener('change', (e) => {
# #             if (e.target.files.length > 0) {
# #                 handleFile(e.target.files[0]);
# #             }
# #         });

# #         function handleFile(file) {
# #             if (!file.type.startsWith('image/')) {
# #                 showError('Please select a valid image file.');
# #                 return;
# #             }

# #             if (file.size > 5 * 1024 * 1024) {
# #                 showError('File size must be less than 5MB.');
# #                 return;
# #             }

# #             selectedFile = file;
            
# #             const reader = new FileReader();
# #             reader.onload = function(e) {
# #                 previewImage.src = e.target.result;
# #                 previewContainer.style.display = 'block';
# #                 processBtn.disabled = false;
                
# #                 uploadArea.innerHTML = `
# #                     <div class="upload-icon">✅</div>
# #                     <div class="upload-text">
# #                         <strong>${file.name}</strong> ready for processing
# #                     </div>
# #                     <div style="color: #888; font-size: 0.9em;">
# #                         Click to select a different image
# #                     </div>
# #                 `;
# #             };
# #             reader.readAsDataURL(file);
            
# #             hideError();
# #         }

# #         async function processImage() {
# #             if (!selectedFile) {
# #                 showError('Please select an image first.');
# #                 return;
# #             }

# #             const method = document.querySelector('input[name="method"]:checked').value;
            
# #             showLoading();
# #             hideError();
# #             hideResults();

# #             try {
# #                 const base64 = await fileToBase64(selectedFile);
                
# #                 const requestData = {
# #                     image: base64.split(',')[1],
# #                     method: method
# #                 };

# #                 const response = await fetch(`${API_BASE}/ocr`, {
# #                     method: 'POST',
# #                     headers: {
# #                         'Content-Type': 'application/json',
# #                     },
# #                     body: JSON.stringify(requestData)
# #                 });

# #                 const data = await response.json();
                
# #                 if (data.success) {
# #                     showResults(data);
# #                 } else {
# #                     throw new Error(data.error || 'OCR processing failed');
# #                 }

# #             } catch (error) {
# #                 console.error('Processing error:', error);
# #                 showError(`Failed to process image: ${error.message}`);
# #             } finally {
# #                 hideLoading();
# #             }
# #         }

# #         function fileToBase64(file) {
# #             return new Promise((resolve, reject) => {
# #                 const reader = new FileReader();
# #                 reader.readAsDataURL(file);
# #                 reader.onload = () => resolve(reader.result);
# #                 reader.onerror = error => reject(error);
# #             });
# #         }

# #         function showResults(data) {
# #             const resultsSection = document.getElementById('results-section');
# #             const resultText = document.getElementById('result-text');
# #             const confidenceBadge = document.getElementById('confidence-badge');

# #             resultText.textContent = data.text || 'No text detected';

# #             const confidence = Math.round((data.confidence || 0) * 100);
# #             confidenceBadge.textContent = `${confidence}% Confidence`;
            
# #             confidenceBadge.className = 'confidence-badge';
# #             if (confidence < 50) {
# #                 confidenceBadge.classList.add('confidence-low');
# #             } else if (confidence < 75) {
# #                 confidenceBadge.classList.add('confidence-medium');
# #             }

# #             resultsSection.style.display = 'block';
# #         }

# #         function showLoading() {
# #             document.getElementById('loading').style.display = 'block';
# #         }

# #         function hideLoading() {
# #             document.getElementById('loading').style.display = 'none';
# #         }

# #         function showError(message) {
# #             const errorElement = document.getElementById('error-message');
# #             errorElement.textContent = message;
# #             errorElement.style.display = 'block';
# #         }

# #         function hideError() {
# #             document.getElementById('error-message').style.display = 'none';
# #         }

# #         function hideResults() {
# #             document.getElementById('results-section').style.display = 'none';
# #         }

# #         async function copyToClipboard() {
# #             const text = document.getElementById('result-text').textContent;
# #             try {
# #                 await navigator.clipboard.writeText(text);
                
# #                 const copyBtn = document.querySelector('.copy-btn');
# #                 const originalText = copyBtn.textContent;
# #                 copyBtn.textContent = '✅ Copied!';
# #                 copyBtn.style.background = '#4CAF50';
                
# #                 setTimeout(() => {
# #                     copyBtn.textContent = originalText;
# #                     copyBtn.style.background = '#4CAF50';
# #                 }, 2000);
# #             } catch (err) {
# #                 showError('Failed to copy text to clipboard');
# #             }
# #         }
# #     </script>
# # </body>
# # </html>
# # """

# # class TrOCRAPIProcessor:
# #     def __init__(self):
# #         self.device = None
# #         self.processor_hw = None
# #         self.model_hw = None
# #         self.processor_pr = None
# #         self.model_pr = None
        
# #         self.tesseract_configs = {
# #             'handwriting_psm6': '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;:()-\'\"',
# #             'handwriting_psm7': '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;:()-\'\"',
# #             'lstm_only': '--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;:()-\'\"\n'
# #         }
        
# #         if TROCR_AVAILABLE:
# #             self.load_models()

# #     def load_models(self):
# #         """Load TrOCR models with error handling"""
# #         try:
# #             logger.info("Loading TrOCR models...")
# #             import torch
# #             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #             logger.info(f"Using device: {self.device}")

# #             # Load models with timeout protection
# #             try:
# #                 self.processor_hw = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
# #                 self.model_hw = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(self.device)
# #                 logger.info("Handwritten model loaded successfully")
# #             except Exception as e:
# #                 logger.error(f"Failed to load handwritten model: {e}")
                
# #             try:
# #                 self.processor_pr = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
# #                 self.model_pr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').to(self.device)
# #                 logger.info("Printed model loaded successfully")
# #             except Exception as e:
# #                 logger.error(f"Failed to load printed model: {e}")

# #         except Exception as e:
# #             logger.error(f"Error during model loading: {e}")
# #             self.device = None

# #     def enhanced_preprocessing_for_trocr(self, image_array):
# #         """Enhanced preprocessing with better error handling"""
# #         try:
# #             # Convert to grayscale if needed
# #             if len(image_array.shape) == 3:
# #                 img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
# #             else:
# #                 img = image_array.copy()
            
# #             # Validate image
# #             if img is None or img.size == 0:
# #                 logger.error("Invalid image provided to preprocessing")
# #                 return None
                
# #             orig_h, orig_w = img.shape

# #             # Multi-stage noise reduction
# #             img = cv2.medianBlur(img, 3)
# #             img = cv2.bilateralFilter(img, 9, 80, 80)
# #             img = cv2.GaussianBlur(img, (3, 3), 0.5)

# #             # Advanced normalization
# #             img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# #             # Contrast enhancement
# #             clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# #             img = clahe.apply(img)

# #             # Multi-threshold approach
# #             _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# #             adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)
# #             adaptive_gauss = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)

# #             # Smart combination
# #             combined = cv2.bitwise_and(otsu, adaptive_mean)
# #             combined = cv2.bitwise_or(combined, adaptive_gauss)

# #             # Morphological operations
# #             kernel_conn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
# #             combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_conn)

# #             # Connected component filtering
# #             num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)
# #             min_area = max(12, (orig_h * orig_w) // 6000)

# #             mask = np.zeros(combined.shape, dtype=np.uint8)
# #             for i in range(1, num_labels):
# #                 area = stats[i, cv2.CC_STAT_AREA]
# #                 height = stats[i, cv2.CC_STAT_HEIGHT]
# #                 width = stats[i, cv2.CC_STAT_WIDTH]
# #                 aspect_ratio = width / height if height > 0 else 0

# #                 if (area >= min_area or
# #                     (area >= 8 and height >= 4 and aspect_ratio < 5) or
# #                     (area >= 4 and height <= 8 and width <= 8)):
# #                     mask[labels == i] = 255

# #             # Skew correction
# #             mask = self.correct_skew_advanced(mask)

# #             # Optimal resizing for TrOCR
# #             height, width = mask.shape
# #             target_height = 160
# #             scale_factor = target_height / height
# #             new_width = int(width * scale_factor)

# #             if new_width > 0 and target_height > 0:
# #                 mask = cv2.resize(mask, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
# #             else:
# #                 logger.warning("Invalid dimensions for resizing, using original")

# #             # Smart padding
# #             pad_h, pad_w = 30, 50
# #             mask = cv2.copyMakeBorder(mask, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=255)

# #             # Ensure proper polarity
# #             if np.mean(mask) < 127:
# #                 mask = cv2.bitwise_not(mask)

# #             return mask

# #         except Exception as e:
# #             logger.error(f"Preprocessing error: {e}")
# #             return image_array if image_array is not None else None

# #     def correct_skew_advanced(self, img):
# #         """Advanced skew correction with improved error handling"""
# #         try:
# #             edges1 = cv2.Canny(img, 30, 100, apertureSize=3)
# #             edges2 = cv2.Canny(img, 50, 150, apertureSize=3)
# #             edges = cv2.bitwise_or(edges1, edges2)

# #             lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)

# #             if lines is not None:
# #                 angles = []
# #                 for line in lines:
# #                     x1, y1, x2, y2 = line[0]
# #                     if abs(x2 - x1) > 10:
# #                         angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
# #                         if -15 < angle < 15:
# #                             angles.append(angle)

# #                 if len(angles) > 5:
# #                     median_angle = np.median(angles)
# #                     if abs(median_angle) > 0.3:
# #                         (h, w) = img.shape[:2]
# #                         center = (w // 2, h // 2)
# #                         M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
# #                         img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# #             return img
# #         except Exception as e:
# #             logger.error(f"Skew correction error: {e}")
# #             return img

# #     def extract_text_with_trocr(self, img):
# #         """TrOCR extraction with improved error handling"""
# #         if img is None or not TROCR_AVAILABLE or self.device is None:
# #             return ""

# #         try:
# #             # Convert to PIL
# #             pil_img = Image.fromarray(img).convert('RGB')

# #             results = []

# #             # Try handwritten model if available
# #             if self.processor_hw is not None and self.model_hw is not None:
# #                 try:
# #                     # Enhanced image for better recognition
# #                     enhancer = ImageEnhance.Contrast(pil_img)
# #                     enhanced = enhancer.enhance(1.3)
# #                     enhancer = ImageEnhance.Sharpness(enhanced)
# #                     enhanced = enhancer.enhance(1.5)

# #                     pixel_values = self.processor_hw(enhanced, return_tensors="pt").pixel_values.to(self.device)
                    
# #                     with torch.no_grad():
# #                         generated_ids = self.model_hw.generate(
# #                             pixel_values,
# #                             max_length=80,
# #                             num_beams=6,
# #                             early_stopping=True,
# #                             do_sample=False,
# #                             length_penalty=0.9,
# #                             repetition_penalty=1.1
# #                         )

# #                     text = self.processor_hw.batch_decode(generated_ids, skip_special_tokens=True)[0]
# #                     if text.strip():
# #                         confidence = self.calculate_enhanced_confidence(text, enhanced, 'handwritten', 6)
# #                         results.append({
# #                             'text': text,
# #                             'confidence': confidence,
# #                             'model': 'handwritten'
# #                         })
# #                 except Exception as e:
# #                     logger.error(f"Handwritten model error: {e}")

# #             # Try printed model if available and handwritten didn't work well
# #             if (self.processor_pr is not None and self.model_pr is not None and 
# #                 (not results or results[0]['confidence'] < 0.7)):
# #                 try:
# #                     pixel_values = self.processor_pr(pil_img, return_tensors="pt").pixel_values.to(self.device)
                    
# #                     with torch.no_grad():
# #                         generated_ids = self.model_pr.generate(
# #                             pixel_values,
# #                             max_length=80,
# #                             num_beams=4,
# #                             early_stopping=True,
# #                             do_sample=False,
# #                             length_penalty=0.9
# #                         )

# #                     text = self.processor_pr.batch_decode(generated_ids, skip_special_tokens=True)[0]
# #                     if text.strip():
# #                         confidence = self.calculate_enhanced_confidence(text, pil_img, 'printed', 4)
# #                         results.append({
# #                             'text': text,
# #                             'confidence': confidence,
# #                             'model': 'printed'
# #                         })
# #                 except Exception as e:
# #                     logger.error(f"Printed model error: {e}")

# #             # Return best result
# #             if results:
# #                 best_result = max(results, key=lambda x: x['confidence'])
# #                 return best_result['text']

# #             return ""

# #         except Exception as e:
# #             logger.error(f"TrOCR extraction error: {e}")
# #             return ""

# #     def calculate_enhanced_confidence(self, text, image, model_type, beam_size):
# #         """Enhanced confidence calculation"""
# #         try:
# #             if not text or not text.strip():
# #                 return 0.0

# #             base_confidence = 0.5
            
# #             # Text quality metrics
# #             valid_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in '.,!?;:()-\'"')
# #             char_ratio = valid_chars / len(text) if text else 0
# #             base_confidence += char_ratio * 0.2

# #             # Length penalty/bonus
# #             if 3 <= len(text.strip()) <= 100:
# #                 base_confidence += 0.1
# #             elif len(text.strip()) > 100:
# #                 base_confidence -= 0.05

# #             # Model-specific adjustments
# #             if model_type == 'handwritten':
# #                 base_confidence += 0.1 if beam_size >= 6 else 0
# #             else:
# #                 base_confidence += 0.05 if beam_size >= 4 else 0

# #             # Word structure bonus
# #             words = text.split()
# #             if words:
# #                 avg_word_len = sum(len(w) for w in words) / len(words)
# #                 if 2 <= avg_word_len <= 8:
# #                     base_confidence += 0.1

# #             return min(max(base_confidence, 0.0), 1.0)

# #         except Exception as e:
# #             logger.error(f"Confidence calculation error: {e}")
# #             return 0.3

# #     def extract_text_with_tesseract(self, img, config_name='handwriting_psm6'):
# #         """Tesseract extraction with multiple configs"""
# #         if not TESSERACT_AVAILABLE or img is None:
# #             return ""

# #         try:
# #             config = self.tesseract_configs.get(config_name, '--oem 3 --psm 6')
            
# #             # Convert to PIL if needed
# #             if isinstance(img, np.ndarray):
# #                 pil_img = Image.fromarray(img)
# #             else:
# #                 pil_img = img

# #             text = pytesseract.image_to_string(pil_img, config=config)
# #             return text.strip()

# #         except Exception as e:
# #             logger.error(f"Tesseract extraction error: {e}")
# #             return ""

# #     def hybrid_extraction(self, img):
# #         """Hybrid approach combining multiple methods"""
# #         results = []

# #         # Try TrOCR if available
# #         if TROCR_AVAILABLE and self.device is not None:
# #             trocr_text = self.extract_text_with_trocr(img)
# #             if trocr_text:
# #                 confidence = self.calculate_enhanced_confidence(trocr_text, img, 'hybrid', 6)
# #                 results.append({
# #                     'text': trocr_text,
# #                     'confidence': confidence,
# #                     'method': 'trocr'
# #                 })

# #         # Try multiple Tesseract configs
# #         if TESSERACT_AVAILABLE:
# #             for config_name in ['handwriting_psm6', 'handwriting_psm7', 'lstm_only']:
# #                 tesseract_text = self.extract_text_with_tesseract(img, config_name)
# #                 if tesseract_text:
# #                     confidence = self.calculate_tesseract_confidence(tesseract_text)
# #                     results.append({
# #                         'text': tesseract_text,
# #                         'confidence': confidence,
# #                         'method': f'tesseract_{config_name}'
# #                     })

# #         # Return best result
# #         if results:
# #             best_result = max(results, key=lambda x: x['confidence'])
# #             return best_result['text'], best_result['confidence']

# #         return "", 0.0

# #     def calculate_tesseract_confidence(self, text):
# #         """Calculate confidence for Tesseract results"""
# #         try:
# #             if not text or not text.strip():
# #                 return 0.0

# #             # Basic quality metrics
# #             base_confidence = 0.4
            
# #             # Character quality
# #             valid_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in '.,!?;:()-\'"')
# #             char_ratio = valid_chars / len(text) if text else 0
# #             base_confidence += char_ratio * 0.3

# #             # Word structure
# #             words = [w for w in text.split() if w.strip()]
# #             if words:
# #                 base_confidence += 0.1

# #             return min(max(base_confidence, 0.0), 1.0)

# #         except Exception as e:
# #             logger.error(f"Tesseract confidence calculation error: {e}")
# #             return 0.3

# # # Initialize processor
# # processor = TrOCRAPIProcessor()

# # @app.route('/')
# # def index():
# #     """Serve the HTML interface"""
# #     return render_template_string(HTML_TEMPLATE)

# # @app.route('/health')
# # def health_check():
# #     """Health check endpoint"""
# #     return jsonify({
# #         'status': 'healthy',
# #         'trocr_available': TROCR_AVAILABLE and processor.device is not None,
# #         'tesseract_available': TESSERACT_AVAILABLE,
# #         'device': str(processor.device) if processor.device else 'CPU'
# #     })

# # @app.route('/ocr', methods=['POST'])
# # def ocr_endpoint():
# #     """Main OCR processing endpoint"""
# #     try:
# #         data = request.get_json()
        
# #         if not data or 'image' not in data:
# #             return jsonify({'success': False, 'error': 'No image data provided'}), 400

# #         # Decode base64 image
# #         try:
# #             image_data = base64.b64decode(data['image'])
# #             image = Image.open(io.BytesIO(image_data))
            
# #             # Convert to RGB if needed
# #             if image.mode != 'RGB':
# #                 image = image.convert('RGB')
                
# #             img_array = np.array(image)
            
# #         except Exception as e:
# #             logger.error(f"Image decode error: {e}")
# #             return jsonify({'success': False, 'error': 'Invalid image data'}), 400

# #         # Get method
# #         method = data.get('method', 'hybrid')
        
# #         # Process image
# #         preprocessed = processor.enhanced_preprocessing_for_trocr(img_array)
# #         if preprocessed is None:
# #             return jsonify({'success': False, 'error': 'Image preprocessing failed'}), 500

# #         # Extract text based on method
# #         if method == 'trocr':
# #             if not TROCR_AVAILABLE or processor.device is None:
# #                 return jsonify({'success': False, 'error': 'TrOCR not available'}), 400
            
# #             text = processor.extract_text_with_trocr(preprocessed)
# #             confidence = processor.calculate_enhanced_confidence(text, preprocessed, 'handwritten', 6)
            
# #         elif method == 'tesseract':
# #             if not TESSERACT_AVAILABLE:
# #                 return jsonify({'success': False, 'error': 'Tesseract not available'}), 400
            
# #             text = processor.extract_text_with_tesseract(preprocessed)
# #             confidence = processor.calculate_tesseract_confidence(text)
            
# #         else:  # hybrid
# #             text, confidence = processor.hybrid_extraction(preprocessed)

# #         return jsonify({
# #             'success': True,
# #             'text': text,
# #             'confidence': confidence,
# #             'method_used': method
# #         })

# #     except Exception as e:
# #         logger.error(f"OCR endpoint error: {e}")
# #         return jsonify({'success': False, 'error': str(e)}), 500

# # @app.errorhandler(413)
# # def too_large(e):
# #     return jsonify({'success': False, 'error': 'File too large'}), 413

# # @app.errorhandler(500)
# # def server_error(e):
# #     return jsonify({'success': False, 'error': 'Internal server error'}), 500

# # if __name__ == '__main__':
# #     # Create temp directory if it doesn't exist
# #     os.makedirs(tempfile.gettempdir(), exist_ok=True)
    
# #     logger.info("Starting OCR API server...")
# #     logger.info(f"TrOCR Available: {TROCR_AVAILABLE}")
# #     logger.info(f"Tesseract Available: {TESSERACT_AVAILABLE}")
    
# #     # Run the Flask app
# #     app.run(
# #         host='0.0.0.0',
# #         port=int(os.environ.get('PORT', 5000)),
# #         debug=os.environ.get('DEBUG', 'False').lower() == 'true',
# #         threaded=True
# #     )

# #!/usr/bin/env python3

from flask import Flask, render_template_string, request, jsonify, send_file
import os
import cv2
import numpy as np
import base64
import io
from PIL import Image, ImageEnhance
import pytesseract
import warnings
warnings.filterwarnings('ignore')

# TrOCR imports
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False

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

    # def calculate_confidence(self, text, image=None):
    #     """Calculate text confidence"""
    #     if not text:
    #         return 0.0
        
    #     text_clean = text.strip()
    #     char_count = len(text_clean)
        
    #     if char_count < 5:
    #         length_score = char_count / 5 * 0.5
    #     elif char_count <= 60:
    #         length_score = 1.0
    #     else:
    #         length_score = max(0.2, 1.0 - (char_count - 60) / 100)
        
    #     alpha_chars = sum(1 for c in text_clean if c.isalpha())
    #     alpha_ratio = alpha_chars / char_count if char_count > 0 else 0
        
    #     words = text_clean.split()
    #     realistic_words = 0
    #     if words:
    #         for word in words:
    #             clean_word = word.strip('.,!?;:()-\'\"').lower()
    #             if len(clean_word) >= 2:
    #                 vowels = sum(1 for c in clean_word if c in 'aeiou')
    #                 if vowels > 0:
    #                     realistic_words += 1
    #         word_realism = realistic_words / len(words)
    #     else:
    #         word_realism = 0
        
    #     final_confidence = (length_score * 0.4 + alpha_ratio * 0.4 + word_realism * 0.2)
    #     return min(1.0, final_confidence)

    def post_process_text(self, text):
        """Post-process OCR text"""
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

    # def extract_text_with_tesseract(self, img):
    #     """Extract text using Tesseract"""
    #     if img is None:
    #         return "", 0.0
        
    #     pil_img = Image.fromarray(img)
    #     enhancer = ImageEnhance.Contrast(pil_img)
    #     pil_img = enhancer.enhance(1.3)
    #     enhancer = ImageEnhance.Sharpness(pil_img)
    #     pil_img = enhancer.enhance(1.4)
        
    #     results = []
    #     for config_name, config in self.tesseract_configs.items():
    #         try:
    #             text = pytesseract.image_to_string(pil_img, config=config)
    #             text = self.post_process_text(text)
    #             if len(text.strip()) > 2:
    #                 confidence = self.calculate_confidence(text)
    #                 results.append({'text': text, 'confidence': confidence})
    #         except:
    #             continue
        
    #     if results:
    #         best_result = max(results, key=lambda x: x['confidence'])
    #         return best_result['text'], best_result['confidence']
        
    #     return "", 0.0

    # def process_image(self, img, method='hybrid'):
    #     """Process image with specified method"""
    #     processed_img, line_images = self.enhanced_preprocessing_multiline(img)
        
    #     if method == 'trocr':
    #         if self.trocr_loaded:
    #             text, confidence = self.extract_text_with_trocr(line_images)
    #         else:
    #             return "TrOCR not available", processed_img, 0.0
                
    #     elif method == 'tesseract':
    #         text, confidence = self.extract_text_with_tesseract(processed_img)
            
    #     else:  # hybrid
    #         trocr_text, trocr_conf = "", 0.0
    #         if self.trocr_loaded:
    #             trocr_text, trocr_conf = self.extract_text_with_trocr(line_images)
            
    #         tesseract_text, tesseract_conf = self.extract_text_with_tesseract(processed_img)
            
    #         if trocr_conf > tesseract_conf * 1.2:
    #             text, confidence = trocr_text, trocr_conf
    #         else:
    #             text, confidence = tesseract_text, tesseract_conf
        
    #     return text, processed_img, confidence

    # def extract_text_with_tesseract(self, img, line_images=None):
    #     """Extract text using Tesseract - Modified to handle multi-line"""
    #     if img is None:
    #         return "", 0.0
        
    #     # If line_images are provided, process each line separately
    #     if line_images and len(line_images) > 0:
    #         line_texts = []
    #         confidences = []
            
    #         for line_img in line_images:
    #             pil_img = Image.fromarray(line_img)
    #             enhancer = ImageEnhance.Contrast(pil_img)
    #             pil_img = enhancer.enhance(1.3)
    #             enhancer = ImageEnhance.Sharpness(pil_img)
    #             pil_img = enhancer.enhance(1.4)
                
    #             best_text = ""
    #             best_confidence = 0.0
                
    #             for config_name, config in self.tesseract_configs.items():
    #                 try:
    #                     text = pytesseract.image_to_string(pil_img, config=config)
    #                     text = self.post_process_text(text)
    #                     if len(text.strip()) > 1:  # More lenient minimum
    #                         confidence = self.calculate_confidence(text)
    #                         if confidence > best_confidence:
    #                             best_text = text
    #                             best_confidence = confidence
    #                 except:
    #                     continue
                
    #             if best_text.strip():
    #                 line_texts.append(best_text.strip())
    #                 confidences.append(best_confidence)
            
    #         final_text = '\n'.join(line_texts) if line_texts else ""
    #         avg_confidence = np.mean(confidences) if confidences else 0.0
    #         return final_text, avg_confidence
        
    #     # Fallback to original single image processing
    #     pil_img = Image.fromarray(img)
    #     enhancer = ImageEnhance.Contrast(pil_img)
    #     pil_img = enhancer.enhance(1.3)
    #     enhancer = ImageEnhance.Sharpness(pil_img)
    #     pil_img = enhancer.enhance(1.4)
        
    #     results = []
    #     for config_name, config in self.tesseract_configs.items():
    #         try:
    #             text = pytesseract.image_to_string(pil_img, config=config)
    #             text = self.post_process_text(text)
    #             if len(text.strip()) > 2:
    #                 confidence = self.calculate_confidence(text)
    #                 results.append({'text': text, 'confidence': confidence})
    #         except:
    #             continue
        
    #     if results:
    #         best_result = max(results, key=lambda x: x['confidence'])
    #         return best_result['text'], best_result['confidence']
        
    #     return "", 0.0

    # def process_image(self, img, method='hybrid'):
    #     """Process image with specified method - Modified for better hybrid logic"""
    #     processed_img, line_images = self.enhanced_preprocessing_multiline(img)
        
    #     if method == 'trocr':
    #         if self.trocr_loaded and line_images:
    #             text, confidence = self.extract_text_with_trocr(line_images)
    #         else:
    #             return "TrOCR not available", processed_img, 0.0
                
    #     elif method == 'tesseract':
    #         # Pass line_images to Tesseract for multi-line processing
    #         text, confidence = self.extract_text_with_tesseract(processed_img, line_images)
            
    #     else:  # hybrid - Modified logic
    #         trocr_text, trocr_conf = "", 0.0
    #         if self.trocr_loaded and line_images:
    #             trocr_text, trocr_conf = self.extract_text_with_trocr(line_images)
            
    #         # Pass line_images to Tesseract for fair comparison
    #         tesseract_text, tesseract_conf = self.extract_text_with_tesseract(processed_img, line_images)
            
    #         # Modified selection logic - more balanced approach
    #         if self.trocr_loaded and trocr_conf > 0.05:  # Lower threshold
    #             # Simple comparison - if TrOCR is significantly better, use it
    #             if trocr_conf > tesseract_conf + 0.1:  # At least 0.1 better
    #                 text, confidence = trocr_text, trocr_conf
    #             elif tesseract_conf > trocr_conf + 0.1:  # Tesseract significantly better
    #                 text, confidence = tesseract_text, tesseract_conf
    #             else:
    #                 # Close confidence - prefer longer meaningful text
    #                 trocr_words = len(trocr_text.split())
    #                 tesseract_words = len(tesseract_text.split())
                    
    #                 if trocr_words > tesseract_words:
    #                     text, confidence = trocr_text, trocr_conf
    #                 else:
    #                     text, confidence = tesseract_text, tesseract_conf
    #         else:
    #             text, confidence = tesseract_text, tesseract_conf
        
    #     return text, processed_img, confidence

    # def calculate_confidence(self, text, image=None):
    #     """Calculate text confidence - Fixed version"""
    #     if not text:
    #         return 0.0
        
    #     text_clean = text.strip()
    #     if not text_clean:
    #         return 0.0
            
    #     char_count = len(text_clean)
        
    #     # Length score
    #     if char_count < 2:
    #         length_score = 0.2
    #     elif char_count <= 30:
    #         length_score = 1.0
    #     elif char_count <= 100:
    #         length_score = 0.9
    #     else:
    #         length_score = max(0.3, 1.0 - (char_count - 100) / 500)
        
    #     # Character analysis
    #     alpha_chars = sum(1 for c in text_clean if c.isalpha())
    #     digit_chars = sum(1 for c in text_clean if c.isdigit())
    #     space_chars = sum(1 for c in text_clean if c.isspace())
    #     punct_chars = sum(1 for c in text_clean if c in '.,!?;:()-\'\"')
        
    #     meaningful_chars = alpha_chars + digit_chars + space_chars + punct_chars
    #     char_ratio = meaningful_chars / char_count if char_count > 0 else 0
        
    #     # Word analysis
    #     words = [w.strip('.,!?;:()-\'\"') for w in text_clean.split() if w.strip()]
    #     word_score = 0.0
        
    #     if words:
    #         valid_words = 0
    #         for word in words:
    #             if len(word) >= 1:
    #                 if word.isdigit():  # Numbers are valid
    #                     valid_words += 1
    #                 elif len(word) == 1 and word.isalpha():  # Single letters OK
    #                     valid_words += 1
    #                 elif len(word) >= 2:
    #                     vowels = sum(1 for c in word.lower() if c in 'aeiou')
    #                     if vowels > 0 or len(word) <= 3:  # Allow short words without vowels
    #                         valid_words += 1
            
    #         word_score = valid_words / len(words)
        
    #     # Final score
    #     final_confidence = (length_score * 0.3 + char_ratio * 0.4 + word_score * 0.3)
    #     return min(1.0, max(0.0, final_confidence))

    def extract_text_with_tesseract(self, img, line_images=None):
        """Extract text using Tesseract - Modified to handle multi-line"""
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

# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Enhanced Multi-Line OCR System</title>
#     <style>
#         * {
#             margin: 0;
#             padding: 0;
#             box-sizing: border-box;
#         }

#         body {
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             min-height: 100vh;
#             padding: 20px;
#         }

#         .container {
#             max-width: 1200px;
#             margin: 0 auto;
#             background: white;
#             border-radius: 20px;
#             box-shadow: 0 20px 40px rgba(0,0,0,0.1);
#             overflow: hidden;
#         }

#         .header {
#             background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
#             color: white;
#             padding: 30px;
#             text-align: center;
#         }

#         .header h1 {
#             font-size: 2.5em;
#             margin-bottom: 10px;
#             font-weight: 300;
#         }

#         .header p {
#             font-size: 1.2em;
#             opacity: 0.9;
#         }

#         .main-content {
#             padding: 40px;
#         }

#         .upload-section {
#             background: #f8f9fa;
#             border-radius: 15px;
#             padding: 30px;
#             margin-bottom: 30px;
#             text-align: center;
#             border: 2px dashed #dee2e6;
#             transition: all 0.3s ease;
#         }

#         .upload-section:hover {
#             border-color: #4ecdc4;
#             background: #f0fffe;
#         }

#         .file-input-wrapper {
#             position: relative;
#             display: inline-block;
#             margin-bottom: 20px;
#         }

#         .file-input {
#             display: none;
#         }

#         .file-input-button {
#             background: linear-gradient(45deg, #4ecdc4, #44a08d);
#             color: white;
#             padding: 15px 30px;
#             border-radius: 50px;
#             border: none;
#             font-size: 1.1em;
#             cursor: pointer;
#             transition: all 0.3s ease;
#             box-shadow: 0 4px 15px rgba(78, 205, 196, 0.3);
#         }

#         .file-input-button:hover {
#             transform: translateY(-2px);
#             box-shadow: 0 6px 20px rgba(78, 205, 196, 0.4);
#         }

#         .options-section {
#             display: flex;
#             gap: 20px;
#             margin-bottom: 30px;
#             flex-wrap: wrap;
#             justify-content: center;
#         }

#         .option-group {
#             background: white;
#             border: 2px solid #e9ecef;
#             border-radius: 10px;
#             padding: 20px;
#             text-align: center;
#             min-width: 150px;
#         }

#         .option-group h3 {
#             color: #495057;
#             margin-bottom: 15px;
#             font-size: 1.1em;
#         }

#         .radio-group {
#             display: flex;
#             flex-direction: column;
#             gap: 10px;
#         }

#         .radio-option {
#             display: flex;
#             align-items: center;
#             gap: 8px;
#             justify-content: center;
#         }

#         .radio-option input[type="radio"] {
#             accent-color: #4ecdc4;
#         }

#         .process-button {
#             background: linear-gradient(45deg, #ff6b6b, #ee5a24);
#             color: white;
#             padding: 15px 40px;
#             border: none;
#             border-radius: 50px;
#             font-size: 1.2em;
#             cursor: pointer;
#             transition: all 0.3s ease;
#             box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
#             display: block;
#             margin: 30px auto;
#         }

#         .process-button:hover:not(:disabled) {
#             transform: translateY(-2px);
#             box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
#         }

#         .process-button:disabled {
#             opacity: 0.6;
#             cursor: not-allowed;
#         }

#         .results-section {
#             display: none;
#             margin-top: 30px;
#         }

#         .results-grid {
#             display: grid;
#             grid-template-columns: 1fr 1fr;
#             gap: 30px;
#             margin-bottom: 30px;
#         }

#         .image-container {
#             background: white;
#             border-radius: 15px;
#             padding: 20px;
#             box-shadow: 0 4px 15px rgba(0,0,0,0.1);
#         }

#         .image-container h3 {
#             color: #495057;
#             margin-bottom: 15px;
#             text-align: center;
#             padding-bottom: 10px;
#             border-bottom: 2px solid #e9ecef;
#         }

#         .image-container img {
#             width: 100%;
#             max-height: 400px;
#             object-fit: contain;
#             border-radius: 10px;
#             box-shadow: 0 2px 10px rgba(0,0,0,0.1);
#         }

#         .text-results {
#             background: white;
#             border-radius: 15px;
#             padding: 20px;
#             box-shadow: 0 4px 15px rgba(0,0,0,0.1);
#         }

#         .text-results h3 {
#             color: #495057;
#             margin-bottom: 15px;
#             display: flex;
#             justify-content: space-between;
#             align-items: center;
#         }

#         .confidence-badge {
#             background: linear-gradient(45deg, #4ecdc4, #44a08d);
#             color: white;
#             padding: 5px 15px;
#             border-radius: 20px;
#             font-size: 0.9em;
#             font-weight: 500;
#         }

#         .text-content {
#             background: #f8f9fa;
#             border-radius: 10px;
#             padding: 20px;
#             font-family: 'Courier New', monospace;
#             font-size: 1.1em;
#             line-height: 1.6;
#             min-height: 150px;
#             white-space: pre-wrap;
#             word-wrap: break-word;
#             border: 1px solid #e9ecef;
#             position: relative;
#         }

#         .copy-button {
#             background: linear-gradient(45deg, #4ecdc4, #44a08d);
#             color: white;
#             border: none;
#             padding: 10px 20px;
#             border-radius: 20px;
#             cursor: pointer;
#             font-size: 0.9em;
#             margin-top: 15px;
#             transition: all 0.3s ease;
#         }

#         .copy-button:hover {
#             transform: translateY(-1px);
#             box-shadow: 0 4px 10px rgba(78, 205, 196, 0.3);
#         }

#         .loading {
#             display: none;
#             text-align: center;
#             padding: 40px;
#         }

#         .spinner {
#             border: 4px solid #f3f3f3;
#             border-top: 4px solid #4ecdc4;
#             border-radius: 50%;
#             width: 40px;
#             height: 40px;
#             animation: spin 1s linear infinite;
#             margin: 0 auto 20px;
#         }

#         @keyframes spin {
#             0% { transform: rotate(0deg); }
#             100% { transform: rotate(360deg); }
#         }

#         .error {
#             background: #ff6b6b;
#             color: white;
#             padding: 15px;
#             border-radius: 10px;
#             margin: 20px 0;
#             display: none;
#         }

#         .success {
#             background: #4ecdc4;
#             color: white;
#             padding: 10px;
#             border-radius: 5px;
#             margin: 10px 0;
#             display: none;
#         }

#         @media (max-width: 768px) {
#             .results-grid {
#                 grid-template-columns: 1fr;
#             }
            
#             .options-section {
#                 flex-direction: column;
#                 align-items: center;
#             }
            
#             .main-content {
#                 padding: 20px;
#             }
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <div class="header">
#             <h1>Enhanced Multi-Line OCR System</h1>
#             <p>Upload an image and extract text using advanced OCR techniques</p>
#         </div>

#         <div class="main-content">
#             <div class="upload-section">
#                 <div class="file-input-wrapper">
#                     <input type="file" id="imageInput" class="file-input" accept="image/*">
#                     <button class="file-input-button" onclick="document.getElementById('imageInput').click()">
#                         📸 Choose Image
#                     </button>
#                 </div>
#                 <p id="fileName" style="color: #6c757d; margin-top: 10px;"></p>
#             </div>

#             <div class="options-section">
#                 <div class="option-group">
#                     <h3>OCR Method</h3>
#                     <div class="radio-group">
#                         <div class="radio-option">
#                             <input type="radio" id="hybrid" name="method" value="hybrid" checked>
#                             <label for="hybrid">Hybrid (Best)</label>
#                         </div>
#                         <div class="radio-option">
#                             <input type="radio" id="trocr" name="method" value="trocr">
#                             <label for="trocr">TrOCR Only</label>
#                         </div>
#                         <div class="radio-option">
#                             <input type="radio" id="tesseract" name="method" value="tesseract">
#                             <label for="tesseract">Tesseract Only</label>
#                         </div>
#                     </div>
#                 </div>
#             </div>

#             <button class="process-button" id="processBtn" onclick="processImage()" disabled>
#                 🚀 Process Image
#             </button>

#             <div class="error" id="errorMsg"></div>
#             <div class="success" id="successMsg"></div>

#             <div class="loading" id="loading">
#                 <div class="spinner"></div>
#                 <p>Processing image... This may take a few moments.</p>
#             </div>

#             <div class="results-section" id="resultsSection">
#                 <div class="results-grid">
#                     <div class="image-container">
#                         <h3>Original Image</h3>
#                         <img id="originalImage" alt="Original Image">
#                     </div>
                    
#                     <div class="image-container">
#                         <h3>Processed Image</h3>
#                         <img id="processedImage" alt="Processed Image">
#                     </div>
#                 </div>

#                 <div class="text-results">
#                     <h3>
#                         Extracted Text
#                         <span class="confidence-badge" id="confidenceBadge">Confidence: 0%</span>
#                     </h3>
#                     <div class="text-content" id="extractedText"></div>
#                     <button class="copy-button" onclick="copyText()">📋 Copy Text</button>
#                 </div>
#             </div>
#         </div>
#     </div>

#     <script>
#         let selectedFile = null;

#         document.getElementById('imageInput').addEventListener('change', function(e) {
#             const file = e.target.files[0];
#             if (file) {
#                 selectedFile = file;
#                 document.getElementById('fileName').textContent = `Selected: ${file.name}`;
#                 document.getElementById('processBtn').disabled = false;
                
#                 // Preview original image
#                 const reader = new FileReader();
#                 reader.onload = function(e) {
#                     document.getElementById('originalImage').src = e.target.result;
#                 };
#                 reader.readAsDataURL(file);
#             }
#         });

#         async function processImage() {
#             if (!selectedFile) {
#                 showError('Please select an image first.');
#                 return;
#             }

#             const method = document.querySelector('input[name="method"]:checked').value;
#             const formData = new FormData();
#             formData.append('image', selectedFile);
#             formData.append('method', method);

#             // Show loading
#             document.getElementById('loading').style.display = 'block';
#             document.getElementById('resultsSection').style.display = 'none';
#             document.getElementById('errorMsg').style.display = 'none';
#             document.getElementById('processBtn').disabled = true;

#             try {
#                 const response = await fetch('/process', {
#                     method: 'POST',
#                     body: formData
#                 });

#                 const result = await response.json();

#                 if (result.success) {
#                     // Display results
#                     document.getElementById('processedImage').src = 'data:image/png;base64,' + result.processed_image;
#                     document.getElementById('extractedText').textContent = result.text || 'No text detected';
#                     document.getElementById('confidenceBadge').textContent = `Confidence: ${Math.round(result.confidence * 100)}%`;
                    
#                     document.getElementById('resultsSection').style.display = 'block';
#                     showSuccess('Image processed successfully!');
#                 } else {
#                     showError(result.error || 'An error occurred during processing.');
#                 }
#             } catch (error) {
#                 console.error('Error:', error);
#                 showError('Network error. Please try again.');
#             } finally {
#                 document.getElementById('loading').style.display = 'none';
#                 document.getElementById('processBtn').disabled = false;
#             }
#         }

#         function copyText() {
#             const textContent = document.getElementById('extractedText').textContent;
#             if (textContent && textContent !== 'No text detected') {
#                 navigator.clipboard.writeText(textContent).then(() => {
#                     showSuccess('Text copied to clipboard!');
#                 }).catch(() => {
#                     showError('Failed to copy text to clipboard.');
#                 });
#             }
#         }

#         function showError(message) {
#             const errorElement = document.getElementById('errorMsg');
#             errorElement.textContent = message;
#             errorElement.style.display = 'block';
#             setTimeout(() => {
#                 errorElement.style.display = 'none';
#             }, 5000);
#         }

#         function showSuccess(message) {
#             const successElement = document.getElementById('successMsg');
#             successElement.textContent = message;
#             successElement.style.display = 'block';
#             setTimeout(() => {
#                 successElement.style.display = 'none';
#             }, 3000);
#         }
#     </script>
# </body>
# </html>
# """

HTML_TEMPLATE="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TextExtract Pro</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='80' font-size='80'>✏️</text></svg>">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #7c3aed 100%);
            min-height: 100vh;
            padding: 20px;
            animation: gradientShift 8s ease-in-out infinite;
        }

        @keyframes gradientShift {
            0%, 100% { background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #7c3aed 100%); }
            50% { background: linear-gradient(135deg, #7c3aed 0%, #dc2626 50%, #ea580c 100%); }
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            box-shadow: 0 32px 64px rgba(0,0,0,0.15);
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header {
            background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
            color: white;
            padding: 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.15"/><circle cx="20" cy="80" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100%" height="100%" fill="url(%23grain)"/></svg>');
            pointer-events: none;
        }

        .header-content {
            position: relative;
            z-index: 1;
        }

        .header h1 {
            font-size: 3.2em;
            margin-bottom: 15px;
            font-weight: 700;
            background: linear-gradient(45deg, #ffffff, #f0f9ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .pen-icon {
            font-size: 1.0em;
            animation: bounce 2s infinite;
            margin-right: 15px;
        }

        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }

        .header p {
            font-size: 1.3em;
            opacity: 0.95;
            font-weight: 300;
            letter-spacing: 0.5px;
        }

        .main-content {
            padding: 50px;
        }

        .upload-section {
            background: linear-gradient(145deg, #f8fafc, #e2e8f0);
            border-radius: 10px;
            padding: 40px;
            margin-bottom: 40px;
            text-align: center;
            border: 3px dashed #1e3a8a; /* Always show dotted border */
            background-clip: padding-box;
            position: relative;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        }

        # .upload-section::before {
        #     content: '';
        #     position: absolute;
        #     top: 0;
        #     left: 0;
        #     right: 0;
        #     bottom: 0;
        #     border-radius: 20px;
        #     padding: 3px;
        #     background: linear-gradient(45deg, #1e3a8a, #3730a3, #7c3aed);
        #     mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        #     mask-composite: exclude;
        #     opacity: 0;
        #     transition: opacity 0.4s ease;
        # }

        .upload-section::before {
            display: none;
        }

        .upload-section:hover::before {
            opacity: 1;
        }

        # .upload-section:hover {
        #     transform: translateY(-2px);
        #     box-shadow: 0 20px 40px rgba(0,0,0,0.12);
        # }

        .upload-section:hover {
            transform: translateY(-2px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.12);
            border-color: #3730a3; /* Change border color on hover */
        }

        .file-input-wrapper {
            position: relative;
            display: inline-block;
            margin-bottom: 25px;
        }

        .file-input {
            display: none;
        }

        # .file-input-button {
        #     background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        #     color: white;
        #     padding: 18px 36px;
        #     border-radius: 50px;
        #     border: none;
        #     font-size: 1.2em;
        #     font-weight: 600;
        #     cursor: pointer;
        #     transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        #     box-shadow: 0 8px 25px rgba(30, 58, 138, 0.4);
        #     letter-spacing: 0.5px;
        #     position: relative;
        #     overflow: hidden;
        # }

        .file-input-button {
            background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
            color: white;
            padding: 20px 36px;
            border-radius: 10px;
            border: none;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 8px 25px rgba(30, 58, 138, 0.4);
            letter-spacing: 0.5px;
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }

        # .file-input-button::before {
        #     content: '';
        #     position: absolute;
        #     top: 0;
        #     left: -100%;
        #     width: 100%;
        #     height: 100%;
        #     background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        #     transition: left 0.6s;
        # }

        .file-input-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            transition: left 0.6s;
        }

        .file-input-button:hover::before {
            left: 100%;
        }

        .file-input-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(30, 58, 138, 0.5);
        }

        .camera-icon {
            font-size: 2em; /* Make icon larger */
            display: block;
        }

        .button-text {
            font-size: 1em;
            display: block;
        }

        .image-preview {
            margin-top: 20px;
            display: none;
        }

        .preview-image {
            max-width: 300px;
            max-height: 200px;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border: 3px solid rgba(255,255,255,0.8);
        }

        .options-section {
            display: flex;
            gap: 30px;
            margin-bottom: 40px;
            flex-wrap: wrap;
            justify-content: center;
        }

        .option-group {
            background: linear-gradient(145deg, #ffffff, #f8fafc);
            border: 2px solid rgba(30, 58, 138, 0.1);
            border-radius: 16px;
            padding: 25px;
            text-align: center;
            min-width: 180px;
            transition: all 0.3s ease;
            box-shadow: 0 8px 25px rgba(0,0,0,0.05);
        }

        .option-group:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 35px rgba(0,0,0,0.1);
            border-color: rgba(30, 58, 138, 0.3);
        }

        .option-group h3 {
            color: #334155;
            margin-bottom: 18px;
            font-size: 1.2em;
            font-weight: 600;
        }

        .radio-group {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .radio-option {
            display: flex;
            align-items: center;
            gap: 10px;
            justify-content: flex-start;
            padding: 8px 12px;
            border-radius: 8px;
            transition: background-color 0.2s ease;
        }

        .radio-option:hover {
            background-color: rgba(30, 58, 138, 0.05);
        }

        .radio-option input[type="radio"] {
            accent-color: #1e3a8a;
            transform: scale(1.2);
        }

        .radio-option label {
            font-weight: 500;
            color: #475569;
        }

        .process-button {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
            padding: 20px 50px;
            border: none;
            border-radius: 50px;
            font-size: 1.3em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 10px 30px rgba(220, 38, 38, 0.4);
            display: block;
            margin: 40px auto;
            letter-spacing: 0.5px;
            position: relative;
            overflow: hidden;
        }

        .process-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            transition: left 0.6s;
        }

        .process-button:hover:not(:disabled)::before {
            left: 100%;
        }

        .process-button:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(220, 38, 38, 0.5);
        }

        .process-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .results-section {
            display: none;
            margin-top: 40px;
            animation: fadeInUp 0.6s ease-out;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            margin-bottom: 40px;
        }

        .image-container {
            background: linear-gradient(145deg, #ffffff, #f8fafc);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 12px 35px rgba(0,0,0,0.08);
            border: 1px solid rgba(30, 58, 138, 0.1);
            transition: all 0.3s ease;
        }

        .image-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 20px 50px rgba(0,0,0,0.12);
        }

        .image-container h3 {
            color: #334155;
            margin-bottom: 20px;
            text-align: center;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(30, 58, 138, 0.1);
            font-size: 1.3em;
            font-weight: 600;
        }

        .image-container img {
            width: 100%;
            max-height: 400px;
            object-fit: contain;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border: 2px solid rgba(255,255,255,0.8);
        }

        .text-results {
            background: linear-gradient(145deg, #ffffff, #f8fafc);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 12px 35px rgba(0,0,0,0.08);
            border: 1px solid rgba(30, 58, 138, 0.1);
        }

        .text-results h3 {
            color: #334155;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 1.3em;
            font-weight: 600;
        }

        .confidence-badge {
            background: linear-gradient(135deg, #1e3a8a, #3730a3);
            color: white;
            padding: 8px 18px;
            border-radius: 25px;
            font-size: 0.9em;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(30, 58, 138, 0.3);
        }

        .text-content {
            background: linear-gradient(145deg, #f8fafc, #e2e8f0);
            border-radius: 12px;
            padding: 25px;
            font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
            font-size: 1.1em;
            line-height: 1.7;
            min-height: 180px;
            white-space: pre-wrap;
            word-wrap: break-word;
            border: 2px solid rgba(30, 58, 138, 0.1);
            position: relative;
            color: #334155;
        }

        .copy-button {
            background: linear-gradient(135deg, #1e3a8a, #3730a3);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            margin-top: 20px;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(30, 58, 138, 0.3);
        }

        .copy-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(30, 58, 138, 0.4);
        }

        .copy-button.copied {
            background: linear-gradient(135deg, #F58F36, #F25F3F);
            box-shadow: 0 6px 20px rgba(5, 150, 105, 0.3);
        }

        .copy-button.copied:hover {
            box-shadow: 0 8px 25px rgba(5, 150, 105, 0.4);
        }

        .loading {
            display: none;
            text-align: center;
            padding: 50px;
        }

        .spinner {
            border: 4px solid rgba(30, 58, 138, 0.1);
            border-top: 4px solid #3730a3;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 25px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading p {
            color: #64748b;
            font-size: 1.1em;
            font-weight: 500;
        }

        .error {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
            padding: 18px 25px;
            border-radius: 12px;
            margin: 25px 0;
            display: none;
            box-shadow: 0 8px 25px rgba(239, 68, 68, 0.3);
            font-weight: 500;
        }

        .success {
            background: linear-gradient(135deg, #3730a3, #3730a3);
            color: white;
            padding: 15px 25px;
            border-radius: 12px;
            margin: 15px 0;
            display: none;
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3);
            font-weight: 500;
        }

        @media (max-width: 768px) {
            .results-grid {
                grid-template-columns: 1fr;
            }
            
            .options-section {
                flex-direction: column;
                align-items: center;
            }
            
            .main-content {
                padding: 30px 25px;
            }

            .header h1 {
                font-size: 2.5em;
            }

            .header p {
                font-size: 1.1em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1>
                    <span class="pen-icon">
                        <svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/>
                        </svg>
                    </span>
                    TextExtract Pro
                </h1>
                <p>Advanced AI-powered OCR system for handwritten and printed text extraction</p>
            </div>
        </div>

        <div class="main-content">
            <div class="upload-section">
                <div class="file-input-wrapper">
                    <input type="file" id="imageInput" class="file-input" accept="image/*">
                    <button class="file-input-button" onclick="document.getElementById('imageInput').click()">
                        <span class="camera-icon">📸</span>
                        <span class="button-text">Choose Image File</span>
                    </button>
                </div>
                <p id="fileName" style="color: #64748b; margin-top: 15px; font-weight: 500;"></p>
                <div class="image-preview" id="imagePreview">
                    <img id="previewImg" class="preview-image" alt="Image Preview">
                </div>
            </div>

            <div class="options-section">
                <div class="option-group">
                    <h3>🤖 OCR Method</h3>
                    <div class="radio-group">
                        <div class="radio-option">
                            <input type="radio" id="hybrid" name="method" value="hybrid" checked>
                            <label for="hybrid">Smart Hybrid</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" id="trocr" name="method" value="trocr">
                            <label for="trocr">TrOCR AI</label>
                        </div>
                        <div class="radio-option">
                            <input type="radio" id="tesseract" name="method" value="tesseract">
                            <label for="tesseract">Tesseract</label>
                        </div>
                    </div>
                </div>
            </div>

            <button class="process-button" id="processBtn" onclick="processImage()" disabled>
                🚀 Extract Text Now
            </button>

            <div class="error" id="errorMsg"></div>
            <div class="success" id="successMsg"></div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing your image with AI... Please wait</p>
            </div>

            <div class="results-section" id="resultsSection">
                <div class="results-grid">
                    <div class="image-container">
                        <h3>📷 Original Image</h3>
                        <img id="originalImage" alt="Original Image">
                    </div>
                    
                    <div class="image-container">
                        <h3>🔧 Processed Image</h3>
                        <img id="processedImage" alt="Processed Image">
                    </div>
                </div>

                <div class="text-results">
                    <h3>
                        📝 Extracted Text
                        <span class="confidence-badge" id="confidenceBadge">Confidence: 0%</span>
                    </h3>
                    <div class="text-content" id="extractedText"></div>
                    <button class="copy-button" id="copyBtn" onclick="copyText()">📋 Copy to Clipboard</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;

        document.getElementById('imageInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Clear previous results when new image is selected
                clearResults();
                
                selectedFile = file;
                document.getElementById('fileName').textContent = `Selected: ${file.name}`;
                document.getElementById('processBtn').disabled = false;
                
                // Show preview of original image
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('previewImg').src = e.target.result;
                    document.getElementById('originalImage').src = e.target.result;
                    document.getElementById('imagePreview').style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });

        function clearResults() {
            document.getElementById('resultsSection').style.display = 'none';
            document.getElementById('errorMsg').style.display = 'none';
            document.getElementById('successMsg').style.display = 'none';
            document.getElementById('extractedText').textContent = '';
            document.getElementById('processedImage').src = '';
            
            // Reset copy button
            const copyBtn = document.getElementById('copyBtn');
            copyBtn.textContent = '📋 Copy to Clipboard';
            copyBtn.classList.remove('copied');
        }

        async function processImage() {
            if (!selectedFile) {
                showError('Please select an image first.');
                return;
            }

            const method = document.querySelector('input[name="method"]:checked').value;
            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('method', method);

            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultsSection').style.display = 'none';
            document.getElementById('errorMsg').style.display = 'none';
            document.getElementById('processBtn').disabled = true;

            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success) {
                    // Display results
                    document.getElementById('processedImage').src = 'data:image/png;base64,' + result.processed_image;
                    document.getElementById('extractedText').textContent = result.text || 'No text detected';
                    document.getElementById('confidenceBadge').textContent = `Confidence: ${Math.round(result.confidence * 100)}%`;
                    
                    document.getElementById('resultsSection').style.display = 'block';
                    showSuccess('Text extraction completed successfully!');
                } else {
                    showError(result.error || 'An error occurred during processing.');
                }
            } catch (error) {
                console.error('Error:', error);
                showError('Network error. Please try again.');
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('processBtn').disabled = false;
            }
        }

        function copyText() {
            const textContent = document.getElementById('extractedText').textContent;
            const copyBtn = document.getElementById('copyBtn');
            
            if (textContent && textContent !== 'No text detected') {
                navigator.clipboard.writeText(textContent).then(() => {
                    // Change button appearance and text
                    copyBtn.textContent = '✅ Copied!';
                    copyBtn.classList.add('copied');
                    
                    showSuccess('Text copied to clipboard successfully!');
                    
                    // Reset button after 3 seconds
                    setTimeout(() => {
                        copyBtn.textContent = '📋 Copy to Clipboard';
                        copyBtn.classList.remove('copied');
                    }, 3000);
                }).catch(() => {
                    showError('Failed to copy text to clipboard.');
                });
            }
        }

        function showError(message) {
            const errorElement = document.getElementById('errorMsg');
            errorElement.textContent = message;
            errorElement.style.display = 'block';
            setTimeout(() => {
                errorElement.style.display = 'none';
            }, 6000);
        }

        function showSuccess(message) {
            const successElement = document.getElementById('successMsg');
            successElement.textContent = message;
            successElement.style.display = 'block';
            setTimeout(() => {
                successElement.style.display = 'none';
            }, 4000);
        }
    </script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

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
        
        return jsonify({
            'success': True,
            'text': text,
            'confidence': confidence,
            'processed_image': processed_img_b64
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'trocr_available': ocr_processor.trocr_loaded
    })

if __name__ == '__main__':
    print("Starting Enhanced Multi-Line OCR System...")
    print(f"TrOCR Available: {ocr_processor.trocr_loaded}")
    app.run(debug=True, host='0.0.0.0', port=5000)

# from flask import Flask, render_template_string, request, jsonify, send_file
# import os
# import cv2
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from collections import Counter
# import pickle
# from scipy import ndimage
# from skimage import measure, morphology, filters
# from skimage.filters import gaussian, unsharp_mask, threshold_otsu, threshold_local
# from skimage.restoration import denoise_nl_means
# import pytesseract
# from PIL import Image, ImageEnhance, ImageFilter, ImageOps
# import warnings
# import base64
# import io
# import json
# from werkzeug.utils import secure_filename
# warnings.filterwarnings('ignore')

# # TrOCR imports
# try:
#     from transformers import TrOCRProcessor, VisionEncoderDecoderModel
#     import torch
#     TROCR_AVAILABLE = True
#     print("TrOCR libraries loaded successfully!")
# except ImportError as e:
#     print(f"TrOCR not available: {e}")
#     TROCR_AVAILABLE = False

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# # Create upload directory
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# class EnhancedMultiLineTrOCRRecognizer:
#     def __init__(self):
#         if TROCR_AVAILABLE:
#             print("Loading TrOCR models...")
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             self.processor_hw = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
#             self.model_hw = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(self.device)
#             self.processor_pr = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
#             self.model_pr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').to(self.device)
#             print("TrOCR models loaded successfully!")

#         self.tesseract_configs = {
#             'multiline_auto': '--oem 3 --psm 3',
#             'multiline_single_block': '--oem 3 --psm 6',
#             'multiline_single_column': '--oem 3 --psm 4',
#             'handwriting_lines': '--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;:()-\'\"\n'
#         }

#     def detect_text_lines(self, img):
#         """Detect and segment text lines in the image"""
#         try:
#             blurred = cv2.GaussianBlur(img, (3, 3), 0)
#             horizontal_projection = np.sum(blurred < 200, axis=1)
#             kernel = np.ones(5) / 5
#             smooth_projection = np.convolve(horizontal_projection, kernel, mode='same')
            
#             threshold = np.max(smooth_projection) * 0.15
#             line_regions = []
            
#             in_line = False
#             start_y = 0
            
#             for y, value in enumerate(smooth_projection):
#                 if not in_line and value > threshold:
#                     start_y = y
#                     in_line = True
#                 elif in_line and value <= threshold:
#                     if y - start_y > 8:
#                         line_regions.append((max(0, start_y - 3), min(img.shape[0], y + 3)))
#                     in_line = False
            
#             if in_line and len(smooth_projection) - start_y > 8:
#                 line_regions.append((max(0, start_y - 3), img.shape[0]))
            
#             if not line_regions:
#                 line_regions = [(0, img.shape[0])]
            
#             merged_regions = []
#             for start, end in sorted(line_regions):
#                 if merged_regions and start <= merged_regions[-1][1] + 5:
#                     merged_regions[-1] = (merged_regions[-1][0], max(merged_regions[-1][1], end))
#                 else:
#                     merged_regions.append((start, end))
            
#             return merged_regions
            
#         except Exception as e:
#             print(f"Line detection error: {e}")
#             return [(0, img.shape[0])]

#     def enhanced_preprocessing_multiline(self, img_path):
#         """Enhanced preprocessing for multi-line text"""
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             return None, []

#         img = cv2.medianBlur(img, 3)
#         img = cv2.bilateralFilter(img, 9, 80, 80)
#         img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
#         clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
#         img = clahe.apply(img)

#         _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
#         combined = cv2.bitwise_and(otsu, adaptive)

#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
#         combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
#         combined = self.correct_skew_advanced(combined)

#         line_regions = self.detect_text_lines(combined)
#         line_images = []
#         for start_y, end_y in line_regions:
#             line_img = combined[start_y:end_y, :]
#             if line_img.shape[0] < 8:
#                 continue
#             line_img = self.preprocess_single_line(line_img)
#             if line_img is not None:
#                 line_images.append(line_img)

#         return combined, line_images

#     def preprocess_single_line(self, line_img):
#         """Preprocess individual text line for optimal recognition"""
#         try:
#             row_sums = np.sum(line_img == 0, axis=1)
#             non_empty_rows = row_sums > 0
#             if not np.any(non_empty_rows):
#                 return None
                
#             line_img = line_img[non_empty_rows, :]

#             target_height = 64
#             height, width = line_img.shape
#             if height > 0:
#                 scale = target_height / height
#                 new_width = max(32, int(width * scale))
#                 line_img = cv2.resize(line_img, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

#             pad_h, pad_w = 16, 32
#             line_img = cv2.copyMakeBorder(line_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=255)

#             if np.mean(line_img) < 127:
#                 line_img = cv2.bitwise_not(line_img)

#             return line_img
#         except Exception as e:
#             print(f"Line preprocessing error: {e}")
#             return None

#     def correct_skew_advanced(self, img):
#         """Advanced skew correction"""
#         try:
#             edges = cv2.Canny(img, 30, 100, apertureSize=3)
#             lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)

#             if lines is not None:
#                 angles = []
#                 for line in lines:
#                     x1, y1, x2, y2 = line[0]
#                     if abs(x2 - x1) > 10:
#                         angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
#                         if -15 < angle < 15:
#                             angles.append(angle)

#                 if len(angles) > 3:
#                     median_angle = np.median(angles)
#                     if abs(median_angle) > 0.3:
#                         (h, w) = img.shape[:2]
#                         center = (w // 2, h // 2)
#                         M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
#                         img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#             return img
#         except:
#             return img

#     def extract_text_from_line_trocr(self, line_img):
#         """Extract text from single line using TrOCR"""
#         if line_img is None or not TROCR_AVAILABLE:
#             return "", 0

#         try:
#             pil_img = Image.fromarray(line_img).convert('RGB')
#             results = []
            
#             enhancer = ImageEnhance.Contrast(pil_img)
#             enhanced1 = enhancer.enhance(1.4)
#             enhancer = ImageEnhance.Sharpness(enhanced1)
#             enhanced1 = enhancer.enhance(1.6)

#             enhancer = ImageEnhance.Contrast(pil_img)
#             enhanced2 = enhancer.enhance(1.2)
#             enhancer = ImageEnhance.Sharpness(enhanced2)
#             enhanced2 = enhancer.enhance(1.3)

#             for strategy_name, enhanced_img in [('high_contrast', enhanced1), ('moderate', enhanced2)]:
#                 for num_beams in [4, 6]:
#                     try:
#                         pixel_values = self.processor_hw(enhanced_img, return_tensors="pt").pixel_values.to(self.device)
#                         with torch.no_grad():
#                             generated_ids = self.model_hw.generate(
#                                 pixel_values,
#                                 max_length=100,
#                                 num_beams=num_beams,
#                                 early_stopping=True,
#                                 do_sample=False,
#                                 length_penalty=0.9,
#                                 repetition_penalty=1.1
#                             )

#                         text = self.processor_hw.batch_decode(generated_ids, skip_special_tokens=True)[0]
#                         if text.strip():
#                             confidence = self.calculate_line_confidence(text, enhanced_img)
#                             results.append({'text': text, 'confidence': confidence})
#                     except:
#                         continue

#             if results:
#                 best_result = max(results, key=lambda x: x['confidence'])
#                 return self.post_process_trocr_text(best_result['text']), best_result['confidence']

#             return "", 0
#         except Exception as e:
#             print(f"Line TrOCR error: {e}")
#             return "", 0

#     def calculate_line_confidence(self, text, image):
#         """Calculate confidence for line text"""
#         if not text:
#             return 0

#         text_clean = text.strip()
#         char_count = len(text_clean)
        
#         if char_count < 2:
#             return 0.1

#         alpha_chars = sum(1 for c in text_clean if c.isalpha())
#         alpha_ratio = alpha_chars / char_count if char_count > 0 else 0
        
#         if 3 <= char_count <= 80:
#             length_score = 1.0
#         else:
#             length_score = max(0.3, 1.0 - abs(char_count - 40) / 50)

#         return min(1.0, alpha_ratio * 0.6 + length_score * 0.4)

#     # def extract_text_multiline_hybrid(self, full_img, line_images):
#     #     """Multi-line hybrid extraction combining TrOCR and Tesseract"""
#     #     if not line_images:
#     #         return "", 0

#     #     line_texts = []
#     #     confidences = []

#     #     for i, line_img in enumerate(line_images):
#     #         trocr_text, trocr_conf = self.extract_text_from_line_trocr(line_img)
            
#     #         if not trocr_text or len(trocr_text.strip()) < 2:
#     #             tesseract_text, tesseract_conf = self.extract_line_tesseract(line_img)
#     #             line_texts.append(tesseract_text if tesseract_text else "")
#     #             confidences.append(tesseract_conf)
#     #         else:
#     #             line_texts.append(trocr_text)
#     #             confidences.append(trocr_conf)

#     #     full_text = ' '.join([text for text in line_texts if text.strip()])
#     #     avg_confidence = np.mean(confidences) if confidences else 0
        
#     #     if not full_text.strip() and full_img is not None:
#     #         full_text, conf = self.extract_full_image_tesseract(full_img)
#     #         avg_confidence = conf

#     #     return full_text.strip(), avg_confidence

#     # Replace the extract_text_multiline_hybrid method with this version:

#     def extract_text_multiline_hybrid(self, full_img, line_images):
#         """Multi-line hybrid extraction combining TrOCR and Tesseract"""
#         if not line_images:
#             return "", 0

#         line_texts = []
#         confidences = []

#         for i, line_img in enumerate(line_images):
#             trocr_text, trocr_conf = self.extract_text_from_line_trocr(line_img)
            
#             if not trocr_text or len(trocr_text.strip()) < 2:
#                 tesseract_text, tesseract_conf = self.extract_line_tesseract(line_img)
#                 line_texts.append(tesseract_text if tesseract_text else "")
#                 confidences.append(tesseract_conf)
#             else:
#                 line_texts.append(trocr_text)
#                 confidences.append(trocr_conf)

#         # Join with newlines instead of spaces to preserve line breaks
#         full_text = '\n'.join([text for text in line_texts if text.strip()])
#         avg_confidence = np.mean(confidences) if confidences else 0
        
#         if not full_text.strip() and full_img is not None:
#             full_text, conf = self.extract_full_image_tesseract(full_img)
#             avg_confidence = conf

#         return full_text.strip(), avg_confidence

#     # def extract_line_tesseract(self, line_img):
#     #     """Extract text from single line using Tesseract"""
#     #     if line_img is None:
#     #         return "", 0

#     #     try:
#     #         pil_img = Image.fromarray(line_img)
            
#     #         configs = [
#     #             '--oem 3 --psm 7',
#     #             '--oem 1 --psm 7',
#     #             '--oem 3 --psm 8'
#     #         ]
            
#     #         best_text = ""
#     #         best_confidence = 0
            
#     #         for config in configs:
#     #             try:
#     #                 text = pytesseract.image_to_string(pil_img, config=config)
#     #                 text = self.clean_ocr_text(text)
                    
#     #                 if len(text.strip()) > 1:
#     #                     confidence = self.assess_text_quality(text)
#     #                     if confidence > best_confidence:
#     #                         best_confidence = confidence
#     #                         best_text = text
#     #             except:
#     #                 continue
                    
#     #         return best_text, best_confidence
#     #     except:
#     #         return "", 0

#     # def extract_full_image_tesseract(self, img):
#     #     """Extract text from full image using Tesseract multiline configs"""
#     #     if img is None:
#     #         return "", 0

#     #     try:
#     #         pil_img = Image.fromarray(img)
#     #         enhancer = ImageEnhance.Contrast(pil_img)
#     #         pil_img = enhancer.enhance(1.3)

#     #         results = []
#     #         for config_name, config in self.tesseract_configs.items():
#     #             try:
#     #                 text = pytesseract.image_to_string(pil_img, config=config)
#     #                 text = self.clean_ocr_text(text)
#     #                 if len(text.strip()) > 2:
#     #                     confidence = self.assess_text_quality(text)
#     #                     results.append({'text': text, 'confidence': confidence})
#     #             except:
#     #                 continue

#     #         if results:
#     #             best_result = max(results, key=lambda x: x['confidence'])
#     #             return best_result['text'], best_result['confidence']
                
#     #         return "", 0
#     #     except:
#     #         return "", 0

#     # def extract_trocr_only(self, full_img, line_images):
#     #     """Extract text using only TrOCR"""
#     #     if not line_images:
#     #         return "", 0

#     #     line_texts = []
#     #     confidences = []

#     #     for line_img in line_images:
#     #         text, conf = self.extract_text_from_line_trocr(line_img)
#     #         if text:
#     #             line_texts.append(text)
#     #             confidences.append(conf)

#     #     full_text = ' '.join(line_texts)
#     #     avg_confidence = np.mean(confidences) if confidences else 0
        
#     #     return full_text.strip(), avg_confidence

#     # Also update the extract_trocr_only method:

#     def extract_trocr_only(self, full_img, line_images):
#         """Extract text using only TrOCR"""
#         if not line_images:
#             return "", 0

#         line_texts = []
#         confidences = []

#         for line_img in line_images:
#             text, conf = self.extract_text_from_line_trocr(line_img)
#             if text:
#                 line_texts.append(text)
#                 confidences.append(conf)

#         # Join with newlines instead of spaces
#         full_text = '\n'.join(line_texts)
#         avg_confidence = np.mean(confidences) if confidences else 0
        
#         return full_text.strip(), avg_confidence

#     # def extract_tesseract_only(self, full_img, line_images):
#     #     """Extract text using only Tesseract"""
#     #     if full_img is not None:
#     #         return self.extract_full_image_tesseract(full_img)
#     #     return "", 0

#     # Update the extract_full_image_tesseract method:

#     def extract_full_image_tesseract(self, img):
#         """Extract text from full image using Tesseract multiline configs"""
#         if img is None:
#             return "", 0

#         try:
#             pil_img = Image.fromarray(img)
#             enhancer = ImageEnhance.Contrast(pil_img)
#             pil_img = enhancer.enhance(1.3)

#             results = []
#             for config_name, config in self.tesseract_configs.items():
#                 try:
#                     text = pytesseract.image_to_string(pil_img, config=config)
#                     # Don't use clean_ocr_text here as it might remove line breaks
#                     text = self.clean_ocr_text_preserve_lines(text)
#                     if len(text.strip()) > 2:
#                         confidence = self.assess_text_quality(text)
#                         results.append({'text': text, 'confidence': confidence})
#                 except:
#                     continue

#             if results:
#                 best_result = max(results, key=lambda x: x['confidence'])
#                 return best_result['text'], best_result['confidence']
                
#             return "", 0
#         except:
#             return "", 0

#     # Add a new method to clean OCR text while preserving line breaks:

#     def clean_ocr_text_preserve_lines(self, text):
#         """Clean OCR text while preserving line breaks"""
#         if not text:
#             return ""
        
#         # Split into lines to process each line separately
#         lines = text.split('\n')
#         cleaned_lines = []
        
#         for line in lines:
#             # Clean each line but preserve the line structure
#             line = ' '.join(line.split())  # Remove extra spaces
            
#             # Apply character corrections while preserving case
#             corrections = {
#                 '|': 'l',
#                 '~': '-', 
#                 'rn': 'm',
#                 '0': 'o',
#                 '1': 'l'
#             }
            
#             for wrong, right in corrections.items():
#                 # Handle lowercase
#                 line = line.replace(wrong, right)
#                 # Handle uppercase  
#                 line = line.replace(wrong.upper(), right.upper())
#                 # Handle title case
#                 line = line.replace(wrong.capitalize(), right.capitalize())
            
#             # Only add non-empty lines
#             if line.strip():
#                 cleaned_lines.append(line.strip())
        
#         # Join lines back with newlines
#         return '\n'.join(cleaned_lines)

#     # Update the extract_tesseract_only method:

#     def extract_tesseract_only(self, full_img, line_images):
#         """Extract text using only Tesseract"""
#         if full_img is not None:
#             return self.extract_full_image_tesseract(full_img)
#         return "", 0

#     # Also update the extract_line_tesseract method to be consistent:

#     def extract_line_tesseract(self, line_img):
#         """Extract text from single line using Tesseract"""
#         if line_img is None:
#             return "", 0

#         try:
#             pil_img = Image.fromarray(line_img)
            
#             configs = [
#                 '--oem 3 --psm 7',
#                 '--oem 1 --psm 7',
#                 '--oem 3 --psm 8'
#             ]
            
#             best_text = ""
#             best_confidence = 0
            
#             for config in configs:
#                 try:
#                     text = pytesseract.image_to_string(pil_img, config=config)
#                     # Use the new cleaning method that preserves case
#                     text = self.clean_ocr_text_preserve_lines(text)
                    
#                     if len(text.strip()) > 1:
#                         confidence = self.assess_text_quality(text)
#                         if confidence > best_confidence:
#                             best_confidence = confidence
#                             best_text = text
#                 except:
#                     continue
                    
#             return best_text, best_confidence
#         except:
#             return "", 0

#     # def post_process_trocr_text(self, text):
#     #     """Enhanced post-processing"""
#     #     if not text:
#     #         return ""

#     #     text = text.strip()
#     #     text = ' '.join(text.split())
        
#     #     text = text.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?')
        
#     #     char_fixes = {'rn': 'm', 'cl': 'd', 'li': 'h', 'vv': 'w', '0': 'o', '1': 'l', '5': 's'}
        
#     #     for wrong, right in char_fixes.items():
#     #         text = text.replace(wrong, right)

#     #     return text.lower().strip()

#     def post_process_trocr_text(self, text):
#         """Enhanced post-processing - preserves original case"""
#         if not text:
#             return ""

#         text = text.strip()
#         text = ' '.join(text.split())
        
#         # Fix spacing around punctuation
#         text = text.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?')
        
#         # Character fixes - but preserve case when possible
#         char_fixes = {
#             'rn': 'm',  # This might need case-sensitive handling
#             'cl': 'd',
#             'li': 'h', 
#             'vv': 'w',
#             '0': 'o',
#             '1': 'l',
#             '5': 's'
#         }
        
#         # Apply fixes while trying to preserve case
#         for wrong, right in char_fixes.items():
#             # Handle lowercase
#             text = text.replace(wrong, right)
#             # Handle uppercase
#             text = text.replace(wrong.upper(), right.upper())
#             # Handle title case
#             text = text.replace(wrong.capitalize(), right.capitalize())

#         return text.strip()  # Removed .lower() to preserve case

#     def assess_text_quality(self, text):
#         """Assess text quality"""
#         if not text or len(text.strip()) < 2:
#             return 0

#         text = text.strip()
#         alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        
#         words = text.split()
#         if not words:
#             return 0.1

#         avg_word_len = sum(len(w.strip('.,!?;:()-\'\"')) for w in words) / len(words)
#         length_score = min(avg_word_len / 5, 1.0)

#         return min(1.0, alpha_ratio * 0.7 + length_score * 0.3)

#     # def clean_ocr_text(self, text):
#     #     """Clean OCR text"""
#     #     text = ' '.join(text.split())
#     #     corrections = {'|': 'l', '~': '-', 'rn': 'm', '0': 'o', '1': 'l'}
#     #     for wrong, right in corrections.items():
#     #         text = text.replace(wrong, right)
#     #     return text.lower().strip()

#     def clean_ocr_text(self, text):
#         """Clean OCR text - preserves original case"""
#         text = ' '.join(text.split())
        
#         corrections = {
#             '|': 'l',
#             '~': '-',
#             'rn': 'm',
#             '0': 'o',
#             '1': 'l'
#         }
        
#         # Apply corrections while preserving case
#         for wrong, right in corrections.items():
#             # Handle lowercase
#             text = text.replace(wrong, right)
#             # Handle uppercase  
#             text = text.replace(wrong.upper(), right.upper())
#             # Handle title case
#             text = text.replace(wrong.capitalize(), right.capitalize())
        
#         return text.strip()  # Removed .lower() to preserve case

#     def process_image(self, img_path, method='hybrid'):
#         """Process single image with specified method"""
#         full_img, line_images = self.enhanced_preprocessing_multiline(img_path)
        
#         if method == 'hybrid':
#             text, confidence = self.extract_text_multiline_hybrid(full_img, line_images)
#         elif method == 'trocr':
#             text, confidence = self.extract_trocr_only(full_img, line_images)
#         elif method == 'tesseract':
#             text, confidence = self.extract_tesseract_only(full_img, line_images)
#         else:
#             text, confidence = self.extract_text_multiline_hybrid(full_img, line_images)
        
#         return {
#             'text': text,
#             'confidence': confidence,
#             'lines_detected': len(line_images),
#             'processed_image': full_img
#         }

# # Initialize recognizer
# recognizer = EnhancedMultiLineTrOCRRecognizer()

# def image_to_base64(img_array):
#     """Convert numpy array to base64 string"""
#     if img_array is None:
#         return None
    
#     # Convert to PIL Image
#     pil_img = Image.fromarray(img_array)
    
#     # Convert to base64
#     buffer = io.BytesIO()
#     pil_img.save(buffer, format='PNG')
#     img_str = base64.b64encode(buffer.getvalue()).decode()
    
#     return f"data:image/png;base64,{img_str}"

# @app.route('/')
# def index():
#     return render_template_string(HTML_TEMPLATE)

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'})
    
#     file = request.files['file']
#     method = request.form.get('method', 'hybrid')
    
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'})
    
#     if file:
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         try:
#             # Process image
#             result = recognizer.process_image(filepath, method)
            
#             # Convert images to base64
#             original_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#             original_b64 = image_to_base64(original_img)
#             processed_b64 = image_to_base64(result['processed_image'])
            
#             response = {
#                 'success': True,
#                 'original_image': original_b64,
#                 'processed_image': processed_b64,
#                 'predicted_text': result['text'],
#                 'confidence': round(result['confidence'] * 100, 2),
#                 'lines_detected': result['lines_detected'],
#                 'method_used': method
#             }
            
#             # Clean up uploaded file
#             os.remove(filepath)
            
#             return jsonify(response)
            
#         except Exception as e:
#             # Clean up uploaded file
#             if os.path.exists(filepath):
#                 os.remove(filepath)
#             return jsonify({'error': f'Processing failed: {str(e)}'})

# HTML_TEMPLATE = '''
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Enhanced Multi-Line TrOCR Text Recognizer</title>
#     <style>
#         * {
#             margin: 0;
#             padding: 0;
#             box-sizing: border-box;
#         }

#         body {
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             min-height: 100vh;
#             padding: 20px;
#         }

#         .container {
#             max-width: 1200px;
#             margin: 0 auto;
#             background: white;
#             border-radius: 20px;
#             box-shadow: 0 20px 40px rgba(0,0,0,0.1);
#             overflow: hidden;
#         }

#         .header {
#             background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
#             color: white;
#             padding: 30px;
#             text-align: center;
#         }

#         .header h1 {
#             font-size: 2.5rem;
#             margin-bottom: 10px;
#             font-weight: 300;
#         }

#         .header p {
#             font-size: 1.1rem;
#             opacity: 0.9;
#         }

#         .main-content {
#             padding: 30px;
#         }

#         .upload-section {
#             background: #f8f9fa;
#             border-radius: 15px;
#             padding: 30px;
#             margin-bottom: 30px;
#             border: 2px dashed #dee2e6;
#             text-align: center;
#             transition: all 0.3s ease;
#         }

#         .upload-section:hover {
#             border-color: #667eea;
#             background: #f0f2ff;
#         }

#         .upload-section.dragover {
#             border-color: #667eea;
#             background: #e8ecff;
#             transform: scale(1.02);
#         }

#         .file-input-wrapper {
#             position: relative;
#             overflow: hidden;
#             display: inline-block;
#         }

#         .file-input {
#             position: absolute;
#             left: -9999px;
#         }

#         .file-input-button {
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#             padding: 15px 30px;
#             border: none;
#             border-radius: 50px;
#             cursor: pointer;
#             font-size: 1.1rem;
#             font-weight: 500;
#             transition: all 0.3s ease;
#             box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
#         }

#         .file-input-button:hover {
#             transform: translateY(-2px);
#             box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
#         }

#         .method-selection {
#             margin: 20px 0;
#         }

#         .method-selection label {
#             font-weight: 600;
#             color: #2c3e50;
#             margin-bottom: 10px;
#             display: block;
#         }

#         .method-buttons {
#             display: flex;
#             gap: 10px;
#             justify-content: center;
#             flex-wrap: wrap;
#         }

#         .method-btn {
#             padding: 10px 20px;
#             border: 2px solid #dee2e6;
#             background: white;
#             border-radius: 25px;
#             cursor: pointer;
#             transition: all 0.3s ease;
#             font-weight: 500;
#         }

#         .method-btn.active {
#             background: #667eea;
#             color: white;
#             border-color: #667eea;
#         }

#         .method-btn:hover {
#             border-color: #667eea;
#             color: #667eea;
#         }

#         .method-btn.active:hover {
#             color: white;
#         }

#         .results-section {
#             display: none;
#             margin-top: 30px;
#         }

#         .results-grid {
#             display: grid;
#             grid-template-columns: 1fr 1fr;
#             gap: 30px;
#             margin-bottom: 30px;
#         }

#         .image-container {
#             background: #f8f9fa;
#             border-radius: 15px;
#             padding: 20px;
#             text-align: center;
#         }

#         .image-container h3 {
#             color: #2c3e50;
#             margin-bottom: 15px;
#             font-weight: 600;
#         }

#         .image-display {
#             max-width: 100%;
#             max-height: 300px;
#             border-radius: 10px;
#             box-shadow: 0 5px 15px rgba(0,0,0,0.1);
#         }

#         .text-results {
#             background: #f8f9fa;
#             border-radius: 15px;
#             padding: 25px;
#             grid-column: 1 / -1;
#         }

#         .text-results h3 {
#             color: #2c3e50;
#             margin-bottom: 20px;
#             font-weight: 600;
#         }

#         # .predicted-text {
#         #     background: white;
#         #     padding: 20px;
#         #     border-radius: 10px;
#         #     border-left: 4px solid #667eea;
#         #     font-size: 1.1rem;
#         #     line-height: 1.6;
#         #     margin-bottom: 20px;
#         #     min-height: 100px;
#         #     font-family: 'Courier New', monospace;
#         # }

        
#         .predicted-text {
#             background: white;
#             padding: 20px;
#             border-radius: 10px;
#             border-left: 4px solid #667eea;
#             font-size: 1.1rem;
#             line-height: 1.6;
#             margin-bottom: 20px;
#             min-height: 100px;
#             font-family: 'Courier New', monospace;
#             white-space: pre-wrap; /* This preserves line breaks and spaces */
#             word-wrap: break-word; /* This handles long words */
#             overflow-wrap: break-word; /* Additional word wrapping support */
#         }

#         .stats-grid {
#             display: grid;
#             grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
#             gap: 15px;
#             margin-bottom: 20px;
#         }

#         .stat-item {
#             background: white;
#             padding: 15px;
#             border-radius: 10px;
#             text-align: center;
#             box-shadow: 0 2px 10px rgba(0,0,0,0.05);
#         }

#         .stat-value {
#             font-size: 1.5rem;
#             font-weight: bold;
#             color: #667eea;
#             margin-bottom: 5px;
#         }

#         .stat-label {
#             font-size: 0.9rem;
#             color: #6c757d;
#             text-transform: uppercase;
#             letter-spacing: 0.5px;
#         }

#         .copy-button {
#             background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
#             color: white;
#             padding: 12px 25px;
#             border: none;
#             border-radius: 25px;
#             cursor: pointer;
#             font-size: 1rem;
#             font-weight: 500;
#             transition: all 0.3s ease;
#             box-shadow: 0 3px 10px rgba(40, 167, 69, 0.3);
#         }

#         .copy-button:hover {
#             transform: translateY(-2px);
#             box-shadow: 0 5px 15px rgba(40, 167, 69, 0.4);
#         }

#         .loading {
#             display: none;
#             text-align: center;
#             padding: 40px;
#         }

#         .spinner {
#             border: 4px solid #f3f3f3;
#             border-top: 4px solid #667eea;
#             border-radius: 50%;
#             width: 50px;
#             height: 50px;
#             animation: spin 1s linear infinite;
#             margin: 0 auto 20px;
#         }

#         @keyframes spin {
#             0% { transform: rotate(0deg); }
#             100% { transform: rotate(360deg); }
#         }

#         .error {
#             background: #f8d7da;
#             color: #721c24;
#             padding: 15px;
#             border-radius: 10px;
#             margin: 20px 0;
#             border-left: 4px solid #dc3545;
#         }

#         .success {
#             background: #d4edda;
#             color: #155724;
#             padding: 15px;
#             border-radius: 10px;
#             margin: 20px 0;
#             border-left: 4px solid #28a745;
#         }

#         @media (max-width: 768px) {
#             .results-grid {
#                 grid-template-columns: 1fr;
#             }
            
#             .method-buttons {
#                 flex-direction: column;
#                 align-items: center;
#             }
            
#             .stats-grid {
#                 grid-template-columns: 1fr;
#             }
            
#             .header h1 {
#                 font-size: 2rem;
#             }
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <div class="header">
#             <h1>Enhanced Multi-Line TrOCR</h1>
#             <p>Advanced Handwritten & Printed Text Recognition System</p>
#         </div>

#         <div class="main-content"><div class="upload-section" id="upload-section">
#                 <div class="file-input-wrapper">
#                     <input type="file" id="file-input" class="file-input" accept="image/*">
#                     <button class="file-input-button" onclick="document.getElementById('file-input').click()">
#                         📁 Choose Image File
#                     </button>
#                 </div>
                
#                 <div class="method-selection">
#                     <label>Recognition Method:</label>
#                     <div class="method-buttons">
#                         <button class="method-btn active" data-method="hybrid">Hybrid (Best)</button>
#                         <button class="method-btn" data-method="trocr">TrOCR Only</button>
#                         <button class="method-btn" data-method="tesseract">Tesseract Only</button>
#                     </div>
#                 </div>
                
#                 <p style="margin-top: 20px; color: #6c757d;">
#                     Drag and drop an image here or click to browse
#                 </p>
#             </div>

#             <div class="loading" id="loading">
#                 <div class="spinner"></div>
#                 <p>Processing your image... This may take a few moments.</p>
#             </div>

#             <div class="results-section" id="results">
#                 <div class="results-grid">
#                     <div class="image-container">
#                         <h3>Original Image</h3>
#                         <img id="original-image" class="image-display" alt="Original Image">
#                     </div>
                    
#                     <div class="image-container">
#                         <h3>Processed Image</h3>
#                         <img id="processed-image" class="image-display" alt="Processed Image">
#                     </div>
#                 </div>

#                 <div class="text-results">
#                     <h3>Recognition Results</h3>
                    
#                     <div class="stats-grid">
#                         <div class="stat-item">
#                             <div class="stat-value" id="confidence-value">0%</div>
#                             <div class="stat-label">Confidence</div>
#                         </div>
#                         <div class="stat-item">
#                             <div class="stat-value" id="lines-value">0</div>
#                             <div class="stat-label">Lines Detected</div>
#                         </div>
#                         <div class="stat-item">
#                             <div class="stat-value" id="method-value">-</div>
#                             <div class="stat-label">Method Used</div>
#                         </div>
#                     </div>

#                     <div class="predicted-text" id="predicted-text">
#                         Your extracted text will appear here...
#                     </div>

#                     <button class="copy-button" onclick="copyText()">
#                         📋 Copy Text to Clipboard
#                     </button>
#                 </div>
#             </div>
#         </div>
#     </div>

#     <script>
#         let selectedMethod = 'hybrid';
#         let extractedText = '';

#         // Method selection
#         document.querySelectorAll('.method-btn').forEach(btn => {
#             btn.addEventListener('click', function() {
#                 document.querySelectorAll('.method-btn').forEach(b => b.classList.remove('active'));
#                 this.classList.add('active');
#                 selectedMethod = this.dataset.method;
#             });
#         });

#         // File input handling
#         const fileInput = document.getElementById('file-input');
#         const uploadSection = document.getElementById('upload-section');
#         const loading = document.getElementById('loading');
#         const results = document.getElementById('results');

#         fileInput.addEventListener('change', function(e) {
#             if (e.target.files.length > 0) {
#                 uploadImage(e.target.files[0]);
#             }
#         });

#         // Drag and drop functionality
#         uploadSection.addEventListener('dragover', function(e) {
#             e.preventDefault();
#             uploadSection.classList.add('dragover');
#         });

#         uploadSection.addEventListener('dragleave', function(e) {
#             e.preventDefault();
#             uploadSection.classList.remove('dragover');
#         });

#         uploadSection.addEventListener('drop', function(e) {
#             e.preventDefault();
#             uploadSection.classList.remove('dragover');
            
#             const files = e.dataTransfer.files;
#             if (files.length > 0 && files[0].type.startsWith('image/')) {
#                 uploadImage(files[0]);
#             } else {
#                 showError('Please drop a valid image file.');
#             }
#         });

#         function uploadImage(file) {
#             const formData = new FormData();
#             formData.append('file', file);
#             formData.append('method', selectedMethod);

#             // Show loading
#             loading.style.display = 'block';
#             results.style.display = 'none';
#             clearErrors();

#             fetch('/upload', {
#                 method: 'POST',
#                 body: formData
#             })
#             .then(response => response.json())
#             .then(data => {
#                 loading.style.display = 'none';
                
#                 if (data.success) {
#                     displayResults(data);
#                 } else {
#                     showError(data.error || 'An error occurred while processing the image.');
#                 }
#             })
#             .catch(error => {
#                 loading.style.display = 'none';
#                 showError('Network error: ' + error.message);
#             });
#         }

#         function displayResults(data) {
#             // Display images
#             document.getElementById('original-image').src = data.original_image;
#             document.getElementById('processed-image').src = data.processed_image;
            
#             // Display text and stats
#             extractedText = data.predicted_text || 'No text detected';
#             document.getElementById('predicted-text').textContent = extractedText;
#             document.getElementById('confidence-value').textContent = data.confidence + '%';
#             document.getElementById('lines-value').textContent = data.lines_detected;
#             document.getElementById('method-value').textContent = data.method_used.toUpperCase();
            
#             // Show results
#             results.style.display = 'block';
            
#             // Show success message
#             showSuccess('Image processed successfully!');
#         }

#         function copyText() {
#             if (extractedText) {
#                 navigator.clipboard.writeText(extractedText).then(function() {
#                     showSuccess('Text copied to clipboard!');
#                 }).catch(function(err) {
#                     // Fallback for older browsers
#                     const textArea = document.createElement('textarea');
#                     textArea.value = extractedText;
#                     document.body.appendChild(textArea);
#                     textArea.select();
#                     try {
#                         document.execCommand('copy');
#                         showSuccess('Text copied to clipboard!');
#                     } catch (err) {
#                         showError('Failed to copy text to clipboard.');
#                     }
#                     document.body.removeChild(textArea);
#                 });
#             } else {
#                 showError('No text to copy.');
#             }
#         }

#         function showError(message) {
#             clearMessages();
#             const errorDiv = document.createElement('div');
#             errorDiv.className = 'error';
#             errorDiv.textContent = message;
#             document.querySelector('.main-content').insertBefore(errorDiv, document.querySelector('.upload-section'));
            
#             // Auto-remove after 5 seconds
#             setTimeout(() => {
#                 if (errorDiv.parentNode) {
#                     errorDiv.parentNode.removeChild(errorDiv);
#                 }
#             }, 5000);
#         }

#         function showSuccess(message) {
#             clearMessages();
#             const successDiv = document.createElement('div');
#             successDiv.className = 'success';
#             successDiv.textContent = message;
#             document.querySelector('.main-content').insertBefore(successDiv, document.querySelector('.upload-section'));
            
#             // Auto-remove after 3 seconds
#             setTimeout(() => {
#                 if (successDiv.parentNode) {
#                     successDiv.parentNode.removeChild(successDiv);
#                 }
#             }, 3000);
#         }

#         function clearMessages() {
#             document.querySelectorAll('.error, .success').forEach(el => {
#                 if (el.parentNode) {
#                     el.parentNode.removeChild(el);
#                 }
#             });
#         }

#         function clearErrors() {
#             document.querySelectorAll('.error').forEach(el => {
#                 if (el.parentNode) {
#                     el.parentNode.removeChild(el);
#                 }
#             });
#         }

#         // File size validation
#         fileInput.addEventListener('change', function(e) {
#             const file = e.target.files[0];
#             if (file) {
#                 // Check file size (16MB limit)
#                 if (file.size > 16 * 1024 * 1024) {
#                     showError('File size must be less than 16MB.');
#                     fileInput.value = '';
#                     return;
#                 }
                
#                 // Check file type
#                 if (!file.type.startsWith('image/')) {
#                     showError('Please select a valid image file.');
#                     fileInput.value = '';
#                     return;
#                 }
#             }
#         });
#     </script>
# </body>
# </html>
# '''

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)