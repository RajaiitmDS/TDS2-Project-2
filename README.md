# Data Analyst Agent

An AI-powered data analysis API that automatically sources, analyzes, and visualizes data using LLMs and returns formatted JSON responses.

## Features

- **Web API**: REST endpoint that accepts questions and data files
- **Data Scraping**: Automatic Wikipedia table extraction
- **Visualization**: Scatter plots with regression lines 
- **Built-in Analysis**: Fallback methods when AI API is unavailable
- **Multiple Formats**: Supports CSV, JSON, text, and image files

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment (optional):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. Run the application:
```bash
python main.py
```

4. Access the web interface at `http://localhost:5000`

## API Usage

### Endpoint
`POST http://localhost:5000/api/`

### Request Format
Use multipart/form-data with:
- `questions.txt` (required): Text file containing analysis questions
- Additional data files (optional): CSV, JSON, images, etc.

### Example with curl
```bash
curl -X POST \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv" \
  http://localhost:5000/api/
```

### Example Questions File
```text
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes.
```

### Response Format
Returns JSON array for numbered questions:
```json
[1, "Titanic", 0.485782, "data:image/png;base64,iVBORw0KG..."]
```

## Project Structure

```
├── main.py              # Application entry point
├── app.py               # Flask application setup
├── data_analyst.py      # Core analysis logic
├── web_scraper.py       # Wikipedia scraping functions
├── visualization.py     # Chart generation
├── templates/
│   └── index.html       # Web interface
├── static/
│   └── style.css        # Styling
├── test_api.py          # Basic API tests
├── evaluation_test.py   # Evaluation format tests
└── requirements.txt     # Python dependencies
```

## Configuration

### AI Proxy Setup
The application is configured to use AI Proxy (for IITM students):
- Base URL: `https://aiproxy.sanand.workers.dev/openai/v1`
- Model: `gpt-4o-mini`
- Fallback to built-in analysis if API unavailable

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key or AI Proxy token
- `SESSION_SECRET`: Flask session secret (optional)

## Built-in Analysis

When AI API is unavailable, the system uses built-in methods for:
- Wikipedia movie data analysis
- Correlation calculations
- Visualization generation
- Data parsing and cleaning

## Testing

Run the evaluation test to verify format compliance:
```bash
python evaluation_test.py
```

Run basic API tests:
```bash
python test_api.py
```

## Deployment

For production deployment:
1. Set proper environment variables
2. Use a production WSGI server like Gunicorn
3. Configure reverse proxy (nginx)
4. Set up SSL/TLS certificates

## License

MIT License