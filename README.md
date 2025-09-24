# 🌱 CropGuard – AI-Powered Agricultural Intelligence Platform

A comprehensive Flask-based web application that empowers farmers with advanced AI-driven tools for crop disease detection, soil analysis, financial planning, marketplace access, and agricultural guidance. Built with cutting-edge technologies including YOLOv8, Groq LLMs, OpenWeatherMap, and SerpAPI.

## 🚀 Key Features

### 🔬 **AI-Powered Crop Disease Detection**
- **Multi-Crop Support**: Detects diseases in 6 major crops:
  - 🌾 **Sugarcane**: 12 diseases (Banded Chlorosis, Brown Rust, Mosaic, Pokkah Boeng, RedRot, Smut, etc.)
  - 🍎 **Apple**: 4 conditions (Black Rot, Rust, Scab, Healthy)
  - 🥥 **Coconut**: 5 diseases (Bud Root Dropping, Bud Rot, Gray Leaf Spot, Leaf Rot, Stem Bleeding)
  - 🍇 **Grapes**: 4 conditions (Black Rot, Esca, Spot, Healthy)
  - 🌽 **Jowar (Sorghum)**: 6 diseases (Anthracnose, Grain Molds, Kernel Smut, Head Smut, Rust, Loose Smut)
  - 🍅 **Tomato**: 10 conditions (Bacterial Spot, Early/Late Blight, Leaf Mold, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy)
- **High Accuracy**: YOLOv8 models with confidence scoring
- **Real-time Processing**: Instant disease detection and classification
- **Smart Recommendations**: AI-generated solutions for each detected disease

### 🌍 **Location-Based Soil Analysis**
- **GPS Integration**: Automatic location detection using coordinates
- **Comprehensive Soil Reports**: Detailed analysis including:
  - pH levels and soil acidity/alkalinity
  - Organic matter content
  - Nitrogen, Phosphorus, and Potassium levels
  - Soil parameter visualization with interactive charts
- **Crop Suitability Assessment**: Recommendations for suitable crops based on soil conditions
- **Fertilizer Recommendations**: Both organic and chemical fertilizer suggestions
- **Soil Management Practices**: Customized advice for soil improvement
- **Visual Reports**: Interactive charts and tables for easy understanding

### 🤖 **Intelligent Agricultural Chatbot**
- **Crop-Specific Knowledge**: Specialized in 6 supported crops (Sugarcane, Jowar, Apple, Coconut, Grapes, Tomato)
- **RAG-Powered Responses**: Retrieval-Augmented Generation using PDF knowledge base
- **Multilingual Support**: Conversational interface in multiple languages
- **Comprehensive Coverage**: 
  - Varietal selection guidance
  - Soil and climate requirements
  - Irrigation and water management
  - Fertilizer schedules and nutrient management
  - Integrated pest and disease management
  - Weed control strategies
  - Harvesting techniques and post-harvest management
  - Government schemes and subsidies
  - Sustainability practices

### 🛒 **Smart Marketplace & Product Recommendations**
- **AI-Validated Products**: Groq-powered validation ensures agriculture-related queries only
- **Real-time Product Search**: Integration with SerpAPI for live product data
- **Comprehensive Product Information**:
  - Product titles and descriptions
  - Current market prices in INR
  - Product images and thumbnails
  - Source verification and links
- **Disease-Specific Recommendations**: Automatic product suggestions based on detected diseases
- **Healthy Plant Maintenance**: Product recommendations for maintaining healthy crops
- **Price Comparison**: Market analysis across different vendors

### 💰 **Advanced Financial Planning**
- **Location-Specific Analysis**: Tailored financial planning based on geographic location
- **Comprehensive Budget Planning**:
  - Detailed expenditure breakdown (fertilizers, seeds, labor, equipment)
  - Expected income projections
  - ROI analysis and profitability assessment
- **Crop-Specific Financial Models**: Customized for each supported crop
- **Land Type Consideration**: Analysis based on soil type and land characteristics
- **Risk Assessment**: Climate and market risk evaluation with mitigation strategies
- **Implementation Timeline**: Step-by-step financial planning with schedules

### 📰 **News & Government Schemes Hub**
- **YouTube Integration**: Embedded video tutorials for agricultural practices
- **Government Scheme Discovery**: AI-powered search for relevant government programs
- **Real-time News**: Latest agricultural news and policy updates
- **Educational Content**: Curated learning resources and best practices
- **Scheme Eligibility**: Information about subsidies and financial assistance programs

### 🌤️ **Weather Intelligence**
- **Real-time Weather Data**: Current temperature, humidity, and weather conditions
- **Location-Based Forecasting**: Weather insights specific to your farm location
- **Agricultural Weather Alerts**: Weather-based farming recommendations

## 🛠️ Technology Stack

### **Backend & Framework**
- **Python 3.10+**: Core programming language
- **Flask**: Lightweight web framework
- **Werkzeug**: Security and file handling utilities

### **AI & Machine Learning**
- **YOLOv8 (Ultralytics)**: Advanced object detection and disease classification
- **Groq LLM**: High-performance language models for intelligent responses
- **ChromaDB**: Vector database for RAG (Retrieval-Augmented Generation)
- **LangChain**: Framework for building LLM applications

### **Data Processing & Visualization**
- **OpenCV**: Computer vision and image processing
- **Matplotlib**: Data visualization and chart generation
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### **External APIs & Services**
- **OpenWeatherMap**: Weather data and geocoding services
- **SerpAPI**: Web search and product discovery
- **Groq API**: Advanced language model inference

### **Frontend & UI**
- **HTML5/CSS3**: Modern web standards
- **Bootstrap**: Responsive design framework
- **JavaScript**: Interactive user interface
- **AJAX**: Asynchronous data communication

## 📁 Project Structure

```
CropGuard/
├── 📄 app.py                          # Main Flask application
├── 🤖 bot.py                          # Agricultural chatbot with RAG
├── 🌱 soil.py                         # Soil analysis and reporting
├── 🛒 product.py                      # Marketplace and product search
├── 📰 News1.py                        # News and government schemes
├── 📋 requirements.txt                # Python dependencies
├── 🔧 .env                           # Environment variables (create this)
│
├── 📁 model/                          # YOLOv8 model files
│   ├── Step-1-yolov8.pt              # Primary disease detection (10MB)
│   ├── Step-2-yolov8.pt              # Detailed classification (31MB)
│   ├── Apple.pt                      # Apple disease detection (10MB)
│   ├── coconut.pt                    # Coconut disease detection (10MB)
│   ├── Grape.pt                      # Grape disease detection (10MB)
│   ├── jowar.pt                      # Jowar disease detection (10MB)
│   └── Tomato.pt                     # Tomato disease detection (10MB)
│
├── 📁 templates/                      # HTML templates
│   ├── index.html                    # Landing page
│   ├── about.html                    # About page
│   ├── chatbot.html                  # Chatbot interface
│   ├── finance.html                  # Financial planning
│   ├── marketplace.html              # Product marketplace
│   ├── news.html                     # News and schemes
│   ├── irrigation_mapper.html        # Irrigation mapping
│   ├── user_dashboard.html           # User dashboard
│   └── [crop].html                   # Individual crop pages
│
├── 📁 static/                         # Static assets
│   ├── css/styles.css                # Custom styling
│   └── images/                       # Images and media
│
├── 📁 data/                          # Knowledge base PDFs
│   └── Apple _merged.pdf             # Apple cultivation guides
│
├── 📁 new_data/                      # Additional knowledge base
│   └── mearged.pdf                   # Merged agricultural resources
│
├── 📁 uploads/                       # User uploaded images
└── 📁 All resources/                 # Comprehensive crop guides
    ├── apple/                        # Apple farming resources
    ├── Coconut/                      # Coconut cultivation guides
    ├── grapes/                       # Grape farming resources
    ├── Jowar/                        # Jowar cultivation guides
    └── tomato/                       # Tomato farming resources
```

## 🔧 Installation & Setup

### **Prerequisites**
- Python 3.10 or higher
- Git (for cloning the repository)
- Virtual environment manager (Conda or venv recommended)

### **Step 1: Clone the Repository**
```bash
git clone <repository-url>
cd "AISSMS FINAL"
```

### **Step 2: Create Virtual Environment**
```bash
# Using Conda (Recommended)
conda create -n cropguard python=3.10 -y
conda activate cropguard

# OR using venv
python -m venv cropguard
source cropguard/bin/activate  # On Windows: cropguard\Scripts\activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Environment Configuration**
Create a `.env` file in the project root with the following variables:

```env
# Flask Configuration
FLASK_SECRET_KEY=your-strong-secret-key-here

# API Keys (Get these from respective providers)
OPENWEATHER_API_KEY=your-openweathermap-api-key
GROQ_API_KEY=your-groq-api-key
SERPAPI_KEY=your-serpapi-key
```

**API Key Setup Guide:**
- **OpenWeatherMap**: Sign up at [openweathermap.org](https://openweathermap.org/api)
- **Groq**: Get API key from [console.groq.com](https://console.groq.com)
- **SerpAPI**: Register at [serpapi.com](https://serpapi.com)

### **Step 5: Model Files Setup**
The YOLOv8 model files are included in the repository and will be automatically loaded when the application starts.

### **Step 6: Run the Application**
```bash
python app.py
```

The application will be available at: `http://localhost:8000`

## 🌐 API Endpoints

### **Disease Detection Endpoints**
- `POST /detects` - Sugarcane disease detection
- `POST /detecta` - Apple disease detection  
- `POST /detectc` - Coconut disease detection
- `POST /detectg` - Grape disease detection
- `POST /detectj` - Jowar disease detection
- `POST /detectt` - Tomato disease detection

### **Location & Weather Services**
- `POST /reverse_geocode` - Convert coordinates to location name
- `POST /update_location` - Get weather data for coordinates

### **Soil Analysis**
- `POST /analyze_soil` - Comprehensive soil analysis with charts and recommendations

### **Financial Planning**
- `POST /analyze_finance` - Agricultural project financial analysis

### **Marketplace & Products**
- `GET/POST /marketplace` - Product search and recommendations

### **News & Information**
- `GET/POST /news` - News articles and government schemes

### **Chatbot Services**
- `GET/POST /chatbot` - AI-powered agricultural assistance

## 🔍 Key Features in Detail

### **Intelligent Disease Detection Workflow**
1. **Image Upload**: Users upload crop images via web interface
2. **Primary Detection**: YOLOv8 model classifies as Healthy/Diseased/Dried
3. **Detailed Analysis**: If diseased, secondary model identifies specific disease
4. **AI Recommendations**: Chatbot provides tailored solutions
5. **Product Suggestions**: Marketplace integration suggests relevant products
6. **Educational Content**: YouTube tutorials and government schemes

### **Soil Analysis Process**
1. **Location Detection**: GPS coordinates converted to location name
2. **Weather Integration**: Current weather data for context
3. **AI Analysis**: Groq LLM analyzes location-specific soil conditions
4. **Parameter Extraction**: pH, nutrients, organic matter analysis
5. **Visualization**: Interactive charts and comparison tables
6. **Recommendations**: Crop suitability and fertilizer suggestions

### **Financial Planning Features**
- **Multi-Factor Analysis**: Location, crop type, land size, budget, soil type
- **Comprehensive Breakdown**: Detailed cost analysis and revenue projections
- **Risk Assessment**: Climate and market risk evaluation
- **Implementation Timeline**: Step-by-step planning with schedules
- **ROI Calculation**: Return on investment analysis

### **Marketplace Intelligence**
- **AI Validation**: Ensures only agriculture-related product searches
- **Real-time Pricing**: Current market prices in Indian Rupees
- **Product Comparison**: Side-by-side product analysis
- **Source Verification**: Trusted vendor information
- **Disease-Specific Recommendations**: Products matched to detected diseases

## 🌍 Multilingual Support

The platform supports multiple languages through:
- **AI-Powered Translation**: Groq LLM handles multilingual queries
- **Localized Responses**: Context-aware responses in user's preferred language
- **Cultural Adaptation**: Farming advice adapted to regional practices
- **Government Scheme Localization**: Region-specific scheme information

## 🔒 Security Features

- **API Key Protection**: All external API keys stored server-side
- **File Upload Security**: Secure filename handling and validation
- **Input Validation**: Comprehensive input sanitization
- **Error Handling**: Graceful error handling with fallback responses

## 🚀 Deployment Options

### **Local Development**
```bash
python app.py
# Runs on http://localhost:8000
```

### **Production Deployment**
```bash
# Using Gunicorn (recommended)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# Using Waitress (Windows)
pip install waitress
waitress-serve --host=0.0.0.0 --port=8000 app:app
```

### **Docker Deployment**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app.py"]
```

## 🛠️ Troubleshooting

### **Common Issues & Solutions**

#### **Port 8000 Already in Use**
```bash
# Change port in app.py
app.run(host="0.0.0.0", port=8001, debug=True)
```

#### **YOLO Model Loading Issues**
- Ensure all `.pt` files are in the `model/` directory
- Check file permissions and disk space
- Verify OpenCV installation: `pip install opencv-python`

#### **API Key Errors (401/403)**
- Verify API keys in `.env` file
- Check API key quotas and billing
- Ensure keys are not expired

#### **ChromaDB Issues**
- ChromaDB creates collections automatically on first run
- Large PDF files may take time for initial vectorization
- Check disk space for vector storage

#### **Weather Data Unavailable**
- Application provides fallback weather data
- Check OpenWeatherMap API key and quota
- Verify internet connectivity

#### **Product Search Issues**
- SerpAPI requires valid subscription
- Check API quota and rate limits
- Verify agriculture-related query validation

### **Performance Optimization**

#### **Model Loading Optimization**
```python
# Load models once at startup (recommended for production)
model = YOLO("model/Step-1-yolov8.pt")
```

#### **Memory Management**
- Monitor memory usage with large PDF files
- Consider model quantization for reduced memory footprint
- Implement caching for frequent queries

## 📊 System Requirements

### **Minimum Requirements**
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **CPU**: Dual-core processor
- **Internet**: Stable connection for API calls

### **Recommended for Production**
- **RAM**: 8GB or more
- **Storage**: 10GB free space
- **CPU**: Quad-core or better
- **GPU**: CUDA-compatible (optional, for faster inference)

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### **Development Guidelines**
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include error handling
- Write tests for new features
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: For the excellent YOLOv8 implementation
- **Groq**: For high-performance LLM inference
- **LangChain**: For the robust LLM application framework
- **OpenWeatherMap**: For reliable weather data
- **SerpAPI**: For comprehensive web search capabilities

## 📞 Support & Contact

For support, feature requests, or questions:
- Create an issue in the repository
- Contact the development team
- Check the troubleshooting section above

---

**Built with ❤️ for the farming community**

*Empowering farmers with AI-driven agricultural intelligence for sustainable and profitable farming.*
