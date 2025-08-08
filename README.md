# CropGuard – AI Agriculture Assistant

A Flask-based web app to help farmers with crop disease detection, soil analysis, weather-aware insights, finance planning, marketplace search, news/schemes, and a chatbot. It uses YOLOv8 models, Groq LLMs, OpenWeatherMap, and SerpAPI.

## Features
- Weather + location dashboard (reverse geocoding; weather panels)
- Soil analysis (Groq) with detailed text, tabular report, and chart
- Crop disease detection (YOLOv8) for sugarcane, jowar, apple, coconut, grape, tomato
- Marketplace search (Groq validation + SerpAPI shopping)
- Finance advisor (Groq) based on location, crop, land size, budget, land type
- News and government schemes helper (Groq + SerpAPI)
- Chatbot focused on the six supported crops (Groq + RAG PDFs)

## Tech Stack
- Backend: Python 3.10+, Flask
- ML: ultralytics YOLOv8 (.pt models in `model/`)
- LLM: Groq (`langchain_groq` via LangChain)
- Vector Database: ChromaDB (embedded, no SQL setup required)
- Data/Vector: Chroma, LangChain, PDFs in `data/` and `new_data/`
- Frontend: HTML/JS (Bootstrap), simple fetch calls

## Project Structure (key paths)
- `app.py`: Flask app, routes, YOLO inference, finance and soil endpoints
- `soil.py`: Soil analysis chain (Groq), parsing, report/graph builders
- `product.py`: Marketplace validation (Groq) + SerpAPI shopping
- `News1.py`: News/schemes helper (Groq) + SerpAPI
- `bot.py`: Crop-focused chatbot (Groq + RAG)
- `templates/`: All pages
- `model/`: YOLOv8 model files (.pt)
- `uploads/`: Uploaded images
- `static/`: CSS/images

## Model Files Setup
The app requires YOLOv8 model files for crop disease detection. These are included in the repository (89MB total).

**Required model files in `model/` directory:**
- `Step-1-yolov8.pt` (10MB) - Initial disease detection
- `Step-2-yolov8.pt` (31MB) - Detailed disease classification  
- `Apple.pt` (10MB) - Apple disease detection
- `coconut.pt` (10MB) - Coconut disease detection
- `Grape.pt` (10MB) - Grape disease detection
- `jowar.pt` (10MB) - Jowar disease detection
- `Tomato.pt` (10MB) - Tomato disease detection


**Note:** The models are tracked with Git LFS for efficient storage and download.

## Database Setup
**No SQL database required!** This app uses:
- **ChromaDB**: An embedded vector database for RAG (Retrieval-Augmented Generation)
- **Local storage**: ChromaDB stores vectors locally in memory/disk
- **Automatic initialization**: ChromaDB creates collections automatically on first run

The app creates two ChromaDB collections:
- `"ollama_embeds"` in `bot.py` (for chatbot RAG over `data/` PDFs)
- `"ollama_embeds"` in `app.py` (for finance analysis over `new_data/` PDFs)

## Environment variables (no hard-coded keys)
Set these variables in a `.env` file in the project root (or in your environment):

```
FLASK_SECRET_KEY=your-strong-secret
OPENWEATHER_API_KEY=your-openweather-key
GROQ_API_KEY=your-groq-key
SERPAPI_KEY=your-serpapi-key
```

Notes:
- Do NOT commit your `.env`.
- On hosted platforms, configure these in dashboard settings.

## Installation
1. Python 3.10+ is recommended. Create a virtual environment (Conda example):
   ```bash
   conda create -n cropguard python=3.10 -y
   conda activate cropguard
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Model files are included in the repository** - no additional download needed
4. Add PDFs to `data/` and `new_data/` (optional but recommended for chatbot/RAG and finance modules).
5. Create `.env` with your keys (see above).

## Running locally
```bash
python app.py
```
The app runs at `http://localhost:8000`.

If port 8000 is busy, change `port=` in `app.run(...)` in `app.py`.

## Key endpoints
- `POST /reverse_geocode` → `{ name }` (server-side reverse geocoding; hides API key)
- `POST /update_location` → `{ weather: { temperature, description, main } }`
- `POST /analyze_soil` → soil report JSON (text + table + base64 chart)
- `POST /marketplace` (form) → product cards via SerpAPI
- `POST /analyze_finance` → finance analysis for `{ location, crop_type, land_size, budget, land_type }`
- `POST /news` (form) → YouTube + Govt schemes links
- `GET/POST /chatbot` → crop-focused chatbot
- YOLO detection routes: `/detects`, `/detecta`, `/detectj`, `/detectc`, `/detectg`, `/detectt`

## Frontend security
- The frontend no longer calls OpenWeatherMap directly.
- Reverse geocoding is proxied through `POST /reverse_geocode` to keep keys server-side.


## Troubleshooting
- macOS port 5000 may be used by AirPlay; this app uses 8000 by default.
- YOLO issues → ensure `.pt` weights exist; OpenCV may need system libs.
- 401 from Groq/SerpAPI/OpenWeather → verify respective API keys.
- If PDFs are large, initial vectorization may take time.
- ChromaDB issues → ensure `chromadb` is installed; collections auto-create on first run.

## License
MIT License