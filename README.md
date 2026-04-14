# India Property AI - Home Price Predictor

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green?style=for-the-badge&logo=flask)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-orange?style=for-the-badge&logo=scikit-learn)

A robust ML-powered platform designed to predict real estate prices with high precision across varying markets. The core of this application runs on a unique **Dual-Brain Architecture** bridging the gap between hyper-localized real estate patterns and general residential markets.

---

## 🧠 The Dual-Brain Architecture

Real estate pricing is largely dictated by location. A single model typically struggles to accurately capture nuances of a hyper-active, high-density metropolitan area alongside generic housing markets.

Our app uses a routing layer to dynamically assign your prediction to one of two specialized expert models:

1. **Bangalore Expert (XGBoost / Gradient Boosting)**: Tuned specifically for Bangalore's dynamic local neighborhoods. It understands localized price levels, BHK availability, and area types like 'Super Built-up' or 'Carpet Area' in fine detail.
2. **General India Expert (Random Forest)**: An overarching model trained to predict housing values by analyzing broad metrics common across any location such as the condition of the home, total property age, garage availability, and floor counts.

---

## ✨ Key Features

- **Price Prediction Hub:** Get instant, high-accuracy pricing brackets whether you drop a pin in a specific Bangalore neighborhood or input generic stats for an unspecified region.
- **Property Upgrade Simulator:** Planning an extension? Want to build a garage or add a balcony? Use the simulator to calculate estimated **Return on Investment (ROI)** by comparing your base property with the simulated upgrades.
- **Market Insights:** Take a peek behind the curtain. Check out visual feature importance metrics to see exactly what drives real estate valuation in our AI's decision process (e.g., Total Area size vs Physical Condition).

---

## 📂 Project Structure

```text
├── models/                         # Pre-trained ML models, scalers, and mapping files
│   ├── bangalore_expert.pkl
│   ├── general_expert.pkl
│   └── ...
├── templates/                      # Dynamic HTML (Flask views) for the web app frontend
│   ├── base.html                   # Master layout with navbar and styling logic
│   ├── home.html                   # Prediction interface
│   ├── insights.html               # ML feature importance visualizer
│   └── simulator.html              # Interactive renovation ROI calculator
├── app.py                          # Core Flask routing, data parsing, and model inference hub
├── bangalore_housing.csv           # Specialized high-density training data
├── House Price Prediction Dataset.csv  # General market training data
└── .gitignore                     
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
Make sure you have Python 3.8+ installed on your system. 

### 2. Clone the repository
```bash
git clone https://github.com/ronitjain7/Home-price-Predictor.git
cd Home-price-Predictor
```

### 3. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost flask
```
*(Note: XGBoost is optional but recommended. If unavailable, the system intelligently falls back to GradientBoostingRegressor).*

### 4. Run the Application
```bash
python app.py
```
This script handles everything! On its first run, it checks if `models/` is uninitialized. If models are missing, it will automatically preprocess the provided CSVs, train both the Bangalore and General experts real-time, scale data metrics, dump the pickles to `models/`, and immediately launch the web server.

### 5. Access the Web App
Open your browser and navigate to:
```text
http://127.0.0.1:5000/
```

---

## 🛠 Tech Stack

- **Backend / Routing**: Python, Flask
- **Machine Learning**: Scikit-Learn, XGBoost, Pandas, Numpy
- **Frontend UI**: Vanilla HTML/Tailwind/CSS, Jinja2 Templates

---

*This robust system is built to minimize bias and maximize precision across the heterogeneous Indian property market.*
