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
- **Gunicorn** - WSGI HTTP server (production)

### Development Tools
- **unittest** - Testing framework
- **pytest** - Advanced testing (optional)

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

## 🔌 API Endpoints Documentation

### Web Routes

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| `GET` | `/` | Home page with file upload form | None |
| `POST` | `/analyze` | Process uploaded dataset | `dataset` (file) |
| `GET` | `/download_stats` | Download statistical summary as CSV | None |

### Request/Response Examples

#### File Upload (`POST /analyze`)
**Request:**
```http
POST /analyze
Content-Type: multipart/form-data

dataset: [file] (CSV/XLS/XLSX)
```

**Response:**
```html
<!-- Returns result.html template with analysis results -->
```

#### Download Stats (`GET /download_stats`)
**Response:**
```csv
column,mean,median,mode,std
sales,200.0000,200.0000,150.0000,70.7107
profit,40.0000,40.0000,30.0000,14.1421
```

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

### Manual Deployment

1. **Prepare for production**
   ```bash
   # Install production dependencies
   pip install gunicorn
   
   # Run with Gunicorn
   gunicorn --bind 0.0.0.0:8000 src.app:app
   ```

2. **Environment setup**
   - Set production environment variables
   - Configure your web server (Nginx, Apache)
   - Set up SSL certificates

### Docker Deployment (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "src.app:app"]
```

## ⚠️ Known Limitations and Future Improvements

### Current Limitations
- **File Size**: Maximum 4MB file size limit (Vercel constraint)
- **File Types**: Only supports CSV, XLS, and XLSX formats
- **Concurrent Users**: Limited by Vercel's serverless architecture
- **Data Privacy**: Files are processed in memory (not stored)
- **AI Dependencies**: Requires OpenAI API key for AI insights

### Planned Future Improvements
- [ ] **Enhanced File Support**: Add support for JSON, Parquet, and other formats
- [ ] **Advanced ML Models**: Integrate scikit-learn for clustering and classification
- [ ] **User Authentication**: Add user accounts and analysis history
- [ ] **Collaborative Features**: Share analyses and add comments
- [ ] **Custom Visualizations**: Allow users to create custom chart types
- [ ] **Data Preprocessing**: Add data cleaning and transformation tools
- [ ] **API Endpoints**: RESTful API for programmatic access
- [ ] **Real-time Collaboration**: Multiple users working on same analysis
- [ ] **Advanced Statistics**: More sophisticated statistical tests
- [ ] **Export Formats**: PDF reports, PowerPoint presentations

### Performance Optimizations
- [ ] **Caching**: Implement Redis for faster repeated analyses
- [ ] **Async Processing**: Background processing for large datasets
- [ ] **CDN Integration**: Faster static asset delivery
- [ ] **Database Integration**: Persistent storage for analysis results

## 📸 Screenshots and Demo

### Live Demo
🔗 **Demo Link**: [Add your Vercel deployment URL here]

### Screenshots
*Screenshots will be added here showing:*
- 📱 **Homepage**: Clean, modern interface with drag-and-drop upload
- 📊 **Analysis Results**: Rich visualizations and AI insights
- 📈 **Charts**: Various chart types including histograms, scatter plots, heatmaps
- 🌙 **Dark Mode**: Beautiful dark theme implementation
- 📱 **Mobile View**: Responsive design on mobile devices

### Video Demo
*[Add link to video demonstration here]*

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ajith Ram**
- GitHub: [@ajithram2003](https://github.com/ajithram2003)
- LinkedIn: [Add your LinkedIn profile]
- Email: [Add your email]

## 🙏 Acknowledgments

- OpenAI for providing the GPT API for AI insights
- The Flask community for excellent documentation
- Vercel for seamless deployment platform
- All contributors and testers

---

⭐ **Star this repository if you found it helpful!**

📧 **Contact**: [Add your contact information]

🐛 **Report Issues**: [GitHub Issues](https://github.com/ajithram2003/Project-AI-Dataset-Insights/issues)
