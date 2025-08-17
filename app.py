import os
import json
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import tempfile
import time
from data_analyst import DataAnalyst

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize data analyst
data_analyst = DataAnalyst()

@app.route('/')
def index():
    """Render the test interface"""
    return render_template('index.html')

@app.route('/api/', methods=['POST'])
def analyze_data():
    """Main API endpoint for data analysis"""
    start_time = time.time()
    
    try:
        logger.info("Received data analysis request")
        
        # Check if questions.txt is present
        if 'questions.txt' not in request.files:
            return jsonify({'error': 'questions.txt file is required'}), 400
        
        questions_file = request.files['questions.txt']
        if questions_file.filename == '':
            return jsonify({'error': 'questions.txt file is required'}), 400
        
        # Read questions
        questions_content = questions_file.read().decode('utf-8')
        logger.info(f"Questions received: {questions_content[:200]}...")
        
        # Process additional files
        uploaded_files = {}
        for file_key in request.files:
            if file_key != 'questions.txt':
                file = request.files[file_key]
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    uploaded_files[file_key] = {
                        'filename': filename,
                        'path': file_path,
                        'original_name': file.filename
                    }
                    logger.info(f"Saved file: {filename}")
        
        # Process the analysis request
        try:
            result = data_analyst.analyze(questions_content, uploaded_files)
            
            # Clean up uploaded files
            for file_info in uploaded_files.values():
                if os.path.exists(file_info['path']):
                    os.remove(file_info['path'])
            
            elapsed_time = time.time() - start_time
            logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            # Clean up uploaded files on error
            for file_info in uploaded_files.values():
                if os.path.exists(file_info['path']):
                    os.remove(file_info['path'])
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        return jsonify({'error': f'Request processing failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500
