# 🏎️ Formula 1 Championship Analysis (1950-2024)

<img src="https://th.bing.com/th/id/OIP.4z9ecvV_YD9lVcVCyGXoogHaEK?w=1600&h=900&rs=1&pid=ImgDetMain" alt="F1 Championship Analysis Dashboard" />

A comprehensive Formula 1 data analysis and prediction platform built with Streamlit and custom machine learning implementations. This application provides historical insights, predictive analytics, and performance clustering for F1 championship data from 1950 to 2024.

## 🌟 Features

### 📈 Championship History
- Interactive visualization of championship points evolution
- Era-based analysis of top performers
- Detailed historical statistics and trends



### 🏁 Race Position Prediction
- Custom Linear Regression implementation
- Grid position and driver-based finish position prediction
- Real-time prediction interface with accuracy metrics



### 🏆 Podium Prediction
- Custom Decision Tree Classification
- Driver performance-based podium finish probability
- Interactive prediction interface

### 📊 Driver Performance Clustering
- Custom K-Means implementation
- Configurable feature selection
- 3D visualization of driver performance clusters
- Detailed cluster analysis

<img src="/api/placeholder/800/400" alt="Driver Performance Clusters" />

### 🏎️ F1 Car Classification
- Custom image classification model
- Support for multiple F1 teams
- Confidence distribution visualization
- Detailed team information display

### 👥 Driver Performance Classification
- Custom K-Nearest Neighbors implementation
- Performance-based driver categorization
- Interactive confusion matrix
- Standardized feature visualization

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express, Plotly Graph Objects
- **Machine Learning**: Custom implementations of:
  - Linear Regression
  - Logistic Regression
  - K-Means Clustering
  - K-Nearest Neighbors
  - Decision Trees
  - Standard Scaler
  - Label Encoder

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/f1-championship-analysis.git
cd f1-championship-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Prepare the data:
   - Create a `data` directory
   - Add the required CSV files:
     - races.csv
     - results.csv
     - drivers.csv
     - constructors.csv
     - qualifying.csv

4. Run the application:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
f1-championship-analysis/
├── app.py                 # Main application file
├── models/               # Custom ML model implementations
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── kmeans.py
│   ├── knn.py
│   └── decision_tree.py
├── utils/               # Utility functions
│   ├── standard_scaler.py
│   └── label_encoder.py
├── data/               # Dataset directory
│   ├── races.csv
│   ├── results.csv
│   ├── drivers.csv
│   ├── constructors.csv
│   └── qualifying.csv
└── requirements.txt    # Project dependencies
```

## 📊 Data Sources

The application uses Formula 1 championship data from 1950 to 2024, including:
- Race results
- Driver information
- Constructor details
- Qualifying data
- Championship standings

## 🚀 Usage

1. Launch the application
2. Select an analysis type from the sidebar
3. Interact with the various features:
   - Adjust parameters using sliders and dropdowns
   - View visualizations
   - Make predictions
   - Explore historical data

## 📋 Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Plotly
- PIL (Python Imaging Library)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Formula 1 for the historical data
- The Streamlit team for the amazing framework
- All contributors and supporters of the project

## 📧 Contact

Your Name - pk6122004@gmail.com

Project Link: [https://github.com/yourusername/f1-championship-analysis](https://github.com/yourusername/f1-championship-analysis)
