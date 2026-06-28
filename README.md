[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20998025.svg)](https://doi.org/10.5281/zenodo.20998025)

# Bangladesh Flood Early Warning System

A machine learning-powered web application for real-time flood risk prediction across Bangladesh, integrating live weather data with trained hydrological models.

## Live Demo
[https://flood-risk-app-7vb5.onrender.com](https://flood-risk-app-7vb5.onrender.com)

## Overview
This system predicts flood risk levels for specific areas in Bangladesh by combining:
- **Live weather data** from OpenWeatherMap API (rainfall, temperature, humidity)
- **Pre-trained Random Forest classifier** on hydrological datasets
- **River level data** from curated area-specific datasets

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Backend | Python 3, Flask |
| ML Model | Random Forest Classifier (scikit-learn) |
| Data Processing | Pandas, NumPy, Joblib |
| External API | OpenWeatherMap REST API |
| Preprocessing | StandardScaler (sklearn) |
| Deployment | Render (Cloud), dotenv |

## Project Structure
    flood-risk-app/
    app.py                         Flask web app and prediction routes
    model_train.py                 Random Forest training pipeline
    Environmental_AI_Project.ipynb Research notebook and EDA
    requirements.txt               Python dependencies
    data/areas.json                Bangladesh area and river level dataset
    models/flood_model.pkl         Trained Random Forest classifier
    models/scaler.pkl              StandardScaler for normalization
    models/flood_area.py           Area-based flood zone definitions
    templates/index.html           Frontend UI

## How It Works
1. User selects a Bangladesh area from the dropdown
2. App fetches live weather data via OpenWeatherMap API
3. Features are scaled and passed to the Random Forest classifier
4. System outputs risk level High or Low with probability percentage

## Local Setup
    git clone https://github.com/yourusername/flood-risk-app
    cd flood-risk-app
    pip install -r requirements.txt
    python app.py

## Research Context
Developed as a pre-PhD research prototype for work at Tulane University Hydroinformatics Lab (River-Coastal Science and Engineering, Aug 2026) under Prof. Dr. Ibrahim Demir, focusing on AI-driven flood risk prediction and early warning systems for climate-vulnerable regions.
