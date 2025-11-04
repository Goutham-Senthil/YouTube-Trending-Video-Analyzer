# YouTube Trending Video Analyzer

A comprehensive project for analyzing YouTube trending videos, performing exploratory data analysis (EDA), building predictive models for view counts, and deploying an interactive Streamlit web application for view prediction.

## Features

- **Data Analysis**: In-depth EDA on YouTube trending video dataset including correlation analysis, channel performance, country-wise views, sentiment analysis, and more.
- **Predictive Modeling**: XGBoost regression model to predict video views based on likes, dislikes, comment count, and title length.
- **Interactive Web App**: Streamlit application for real-time view prediction using user inputs.
- **Visualization**: Various plots and charts for insights into trending patterns, word clouds, and feature importance.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd YouTube-Trending-Video-Analyzer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit App
To launch the interactive view prediction app:
```bash
streamlit run app.py
```
This will open a web browser where you can input video metrics (likes, dislikes, comment count, title length) and get predicted view counts.

### Running the Jupyter Notebook
Open `main.ipynb` in Jupyter Notebook or JupyterLab to explore the data analysis, model training, and evaluation process. The notebook includes:
- Data cleaning and preprocessing
- Exploratory data analysis
- Model building and hyperparameter tuning
- Feature importance analysis

## Dataset

The project uses a YouTube trending videos dataset (`dataset/youtube.csv`) obtained from [Kaggle](https://www.kaggle.com/datasets/thedevastator/youtube-trending-videos-dataset). It contains information about trending videos across different countries. Key columns include:
- Video metadata (title, channel, publish date, etc.)
- Engagement metrics (views, likes, dislikes, comments)
- Categorical features (country, tags, etc.)

## Model

The predictive model is an XGBoost regressor trained to predict log-transformed view counts. The model uses features like likes, dislikes, comment count, and title length. The trained model is saved as `youtube_trending_model.json` for use in the Streamlit app.

## Requirements

- Python 3.7+
- Libraries listed in `requirements.txt`:
  - streamlit
  - xgboost
  - pandas
  - numpy
  - scikit-learn

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
