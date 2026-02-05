📈 Stockprediction

A Python-based stock market prediction application that combines historical price trends with real-time news sentiment analysis to generate more informed stock movement predictions.
The system enhances traditional time-series forecasting by incorporating market sentiment extracted from financial news using VADER, allowing predictions to better reflect real-world market psychology.

🧠 Project Overview

Stock prices are influenced not only by historical data but also by investor sentiment and market news. This project addresses that gap by blending:

📊 Historical stock price trends

📰 Real-time financial news

😊 Sentiment polarity analysis (VADER)

By correlating sentiment scores with price movement, the model attempts to validate and strengthen trend predictions.

🚀 Features

✔️ Fetches historical stock data

✔️ Predicts future trends using statistical / ML models

✔️ Integrates News API to fetch real-time financial news

✔️ Applies VADER Sentiment Analysis to quantify news impact

✔️ Matches sentiment trends with price movement

✔️ Extendable architecture for adding advanced ML/DL models

✔️ Web-based interface for visualization (if enabled)

🛠️ Tech Stack

        Technology--------------Purpose
        Python------------------Core logic and modeling
        Yahoo Finance Dataset---Historical stock prices
        News API----------------Real-time financial news extraction
        VADER (NLTK)------------Sentiment analysis of news headlines
        Pandas / NumPy----------Data processing
        Flask (if applicable)---Web interface
        HTML / CSS / JS---------Frontend visualization

📦 Repository Structure

📦 Stockprediction

        ├── static/---------------------# CSS, JS, assets
        ├── templates/------------------# HTML templates
        ├── constants.py----------------# API keys & configuration
        ├── main.py---------------------# Core prediction pipeline
        ├── update_nse_trends.py--------# Stock & trend updates
        ├── requirements.txt------------# Dependencies
        ├── Procfile--------------------# Deployment configuration
        ├── Yahoo-Finance-Ticker-Symbols.csv
        └── logs / outputs--------------# Prediction & sentiment logs

⚙️ How It Works

Historical Data Collection

-Stock price data is fetched from Yahoo Finance datasets.

Trend Analysis

-Time-series analysis is applied to identify upward or downward movement.

News Fetching (News API)

-Recent financial news related to selected stocks is collected using News API.

Sentiment Analysis (VADER)

-News headlines are analyzed using VADER

-Each headline is assigned a positive, negative, or neutral score

-An overall sentiment score is calculated

Trend Matching

-Sentiment trends are compared with price trends to:

--Validate predictions

--Detect possible reversals

--Strengthen confidence in forecasts

Prediction Output

-Final trend prediction is generated and logged / displayed.

🧠 Why VADER?

Designed specifically for short text (headlines, tweets, news)

Works well without heavy training

Captures market emotion, polarity, and intensity

Lightweight and fast for real-time sentiment scoring

📊 Example Use Case

📉 Stock shows a downward trend

📰 News sentiment turns strongly positive

👉 Potential trend reversal signal

This hybrid approach provides context-aware predictions instead of relying solely on historical prices.

📸 Application Screenshots & Visual Insights

📊 7-Day Price Target Forecast

<img width="751" height="736" alt="image" src="https://github.com/user-attachments/assets/e2bc2c7c-601f-4d8b-a237-ba5dcd43b6ba" />

<img width="620" height="667" alt="image" src="https://github.com/user-attachments/assets/ac157528-2d80-45bc-b04a-72c82e89b34b" />

📰 News Sentiment Analysis (VADER + News API)<img width="933" height="615" alt="image" src="https://github.com/user-attachments/assets/cf355f27-e4e9-465c-9c07-7e4e3a534eb3" />

📈 ARIMA Price Forecast vs Actual<img width="1394" height="405" alt="image" src="https://github.com/user-attachments/assets/793c5d7a-8590-4b87-980f-b9784a13396e" />

🤖 Multi-Model Forecast Comparison<img width="1413" height="349" alt="image" src="https://github.com/user-attachments/assets/539e78a3-f7fd-408b-a26a-46d4c5fdf3dc" />

📉 Technical Indicators Dashboard<img width="1218" height="594" alt="image" src="https://github.com/user-attachments/assets/33a7a11d-4037-4427-90f4-a7f89c6bb90f" />

▶️ Run the Application

 1) Clone the Repo
    
        git clone https://github.com/VenCasMet/Stockprediction.git
        cd Stockprediction
    
2) Create & Activate Virtual Environment
   
        python3 -m venv venv
        source venv/bin/activate      # macOS/Linux
        venv\Scripts\activate         # Windows

3) Install Dependencies

        pip install -r requirements.txt

After completing the setup and installing dependencies, start the application using: 

        python main.py

Once running, open your browser and access the dashboard (if enabled) to view stock forecasts, sentiment analysis, and model insights.
