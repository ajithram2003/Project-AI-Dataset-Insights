# AI Dataset Insights Generator

A powerful web application that transforms raw datasets into actionable insights using AI-powered analysis and beautiful visualizations.

## 📊 Project Overview and Objectives

The AI Dataset Insights Generator is designed to help data analysts, researchers, and business professionals quickly understand their data through:

- **Automated Data Analysis**: Statistical summaries, correlation analysis, and outlier detection
- **AI-Powered Insights**: Intelligent recommendations and pattern recognition using OpenAI GPT
- **Rich Visualizations**: Interactive charts including histograms, scatter plots, correlation heatmaps, and more
- **Export Capabilities**: Download statistical summaries and generate printable reports
- **User-Friendly Interface**: Drag-and-drop file upload with real-time progress tracking

### Key Features
- 🧠 **AI-Powered Analysis** - Get intelligent insights and recommendations
- 📈 **Rich Visualizations** - Beautiful charts and graphs to understand your data
- 📁 **Multiple File Formats** - Support for CSV, XLS, and XLSX files
- 🌙 **Dark/Light Mode** - Modern UI with theme switching
- 📱 **Responsive Design** - Works perfectly on desktop and mobile
- ⚡ **Real-time Processing** - Fast analysis with progress indicators
- 📊 **Export & Share** - Download reports and share insights

## 🛠 Tech Stack and Dependencies

### Backend
- **Python 3.8+** - Core programming language
- **Flask 3.0+** - Web framework
- **Pandas 2.2+** - Data manipulation and analysis
- **NumPy 2.0+** - Numerical computing
- **Matplotlib 3.8+** - Data visualization
- **OpenAI 1.30+** - AI-powered insights generation

### Frontend
- **HTML5 & CSS3** - Structure and styling
- **JavaScript (ES6+)** - Interactive functionality
- **Pico CSS** - Modern CSS framework
- **Font Awesome** - Icons and visual elements

### Deployment
- **Vercel** - Serverless deployment platform


## 🚀 Setup and Installation Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ajithram2003/Project-AI-Dataset-Insights.git
   cd Project-AI-Dataset-Insights
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create a .env file in the project root
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   echo "FLASK_SECRET_KEY=your_secret_key_here" >> .env
   ```

5. **Run the application**
   ```bash
   python src/app.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

### Running Tests
```bash
# Run all tests
python -m unittest discover tests -v

# Run specific test file
python -m unittest tests.test_app -v
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Home page with file upload form |
| `POST` | `/analyze` | Process uploaded dataset |
| `GET` | `/health` | Health check endpoint |

## 🚀 Deployment Instructions

### Vercel Deployment (Recommended)

1. **Connect to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Sign in with your GitHub account
   - Click "New Project"
   - Import your GitHub repository

2. **Configure Environment Variables**
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   FLASK_SECRET_KEY=your_secret_key_here
   ```

3. **Deploy**
   - Vercel will automatically detect the Python project
   - The `vercel.json` configuration will be used
   - Your app will be deployed and accessible via Vercel URL


## ⚠️ Known Limitations and Future Improvements

### Current Limitations
- **File Size**: Maximum 4MB file size limit (Vercel constraint)
- **File Types**: Only supports CSV, XLS, and XLSX formats
- **Serverless Architecture**: Each request is independent (no session persistence)
- **Data Privacy**: Files are processed in memory (not stored)
- **AI Dependencies**: Requires OpenAI API key for AI insights
- **CSV Download**: Not available in serverless mode (use print function instead)
- **Cold Starts**: First request after inactivity may take 1-3 seconds

## 📸 Screenshots and Demo

### Live Demo
🔗 **Demo Link**: https://project-ai-dataset-insights.vercel.app/

### Screenshots

#### 🏠 Homepage - File Upload Interface
![Homepage](https://github.com/ajithram2003/Project-AI-Dataset-Insights/blob/main/docs/images/homepage.jpeg?raw=true)

*Clean, modern interface with drag-and-drop file upload functionality*

#### 📊 Analysis Results - AI Insights and Statistics
![Analysis Results](https://github.com/ajithram2003/Project-AI-Dataset-Insights/blob/main/docs/images/analysis-results.jpeg?raw=true)

*Comprehensive analysis showing AI-generated insights and statistical summaries*

#### 📈 Data Visualizations Dashboard
![Data Visualizations](https://github.com/ajithram2003/Project-AI-Dataset-Insights/blob/main/docs/images/data-visualizations.jpeg?raw=true)

*Rich collection of charts including bar charts, line charts, histograms, and correlation heatmaps*

### Key Features Showcased:
- 🎨 **Modern UI Design**: Clean, professional interface with gradient themes
- 📁 **Drag & Drop Upload**: Intuitive file upload with progress indicators
- 🧠 **AI-Powered Insights**: Intelligent analysis and recommendations
- 📊 **Rich Visualizations**: Multiple chart types for comprehensive data understanding
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices
- 🌙 **Theme Switching**: Dark/Light mode toggle for user preference


## 👨‍💻 Author

**Ajith Ram**
- GitHub: [@ajithram2003](https://github.com/ajithram2003)
