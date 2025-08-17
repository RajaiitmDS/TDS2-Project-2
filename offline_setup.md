# Offline Setup Instructions

This document provides step-by-step instructions to set up and run the Data Analyst Agent offline.

## Prerequisites

- Python 3.11 or higher
- pip (Python package installer)
- Internet connection for initial setup only

## Installation Steps

### 1. Download All Files

Ensure you have all these files in your project directory:

```
data-analyst-agent/
├── main.py
├── app.py
├── data_analyst.py
├── web_scraper.py
├── visualization.py
├── requirements_offline.txt
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── test_api.py
├── evaluation_test.py
├── README.md
└── offline_setup.md
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements_offline.txt
```

### 4. Run the Application

```bash
python main.py
```

The application will start on `http://localhost:5000`

### 5. Test the Setup

Run the test scripts to verify everything works:

```bash
# Basic API test
python test_api.py

# Evaluation format test
python evaluation_test.py
```

## Configuration

### Environment Variables (Optional)

Create a `.env` file or set environment variables:

```bash
# For OpenAI API (optional)
export OPENAI_API_KEY="your-api-key-here"

# For Flask session (optional)
export SESSION_SECRET="your-secret-key-here"
```

### AI Proxy Configuration

The application is pre-configured for AI Proxy service. If you have access to AI Proxy:

1. Get your token from AI Proxy dashboard
2. Set it as OPENAI_API_KEY environment variable
3. The application will automatically use AI Proxy endpoint

### Fallback Mode

If no API key is provided, the application runs in fallback mode with built-in analysis methods for:
- Wikipedia movie data analysis
- Statistical calculations
- Data visualization

## Usage

### Web Interface

1. Open `http://localhost:5000` in your browser
2. Upload a `questions.txt` file with your analysis questions
3. Optionally upload data files (CSV, JSON, etc.)
4. Click "Analyze" to get results

### API Endpoint

Use the `/api/` endpoint for programmatic access:

```bash
curl -X POST \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv" \
  http://localhost:5000/api/
```

## Troubleshooting

### Common Issues

1. **Module not found errors**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements_offline.txt` again

2. **Port 5000 already in use**
   - Change the port in `main.py`: `app.run(host='0.0.0.0', port=8000, debug=True)`

3. **Permission errors**
   - Run with proper permissions
   - Check file ownership and permissions

4. **API errors**
   - Application falls back to built-in analysis automatically
   - Check logs for detailed error messages

### Debug Mode

The application runs in debug mode by default. Check the console output for detailed error messages and logs.

## Production Deployment

For production use:

1. Set `debug=False` in `main.py`
2. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn --bind 0.0.0.0:5000 main:app
   ```
3. Set up reverse proxy (nginx, Apache)
4. Configure SSL/TLS certificates
5. Set secure environment variables

## File Descriptions

- `main.py`: Application entry point
- `app.py`: Flask application configuration
- `data_analyst.py`: Core analysis logic with AI integration
- `web_scraper.py`: Wikipedia and web scraping functions
- `visualization.py`: Chart generation using matplotlib
- `templates/index.html`: Web interface template
- `static/style.css`: Web interface styling
- `test_api.py`: Basic functionality tests
- `evaluation_test.py`: Evaluation format compliance tests