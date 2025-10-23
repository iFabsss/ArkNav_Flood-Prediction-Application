# ArkNav: An AI-Powered Flood Risk Knowledge and Navigation Application 🌊🚨

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Flutter](https://img.shields.io/badge/flutter-3.13-blue)](https://flutter.dev/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Table of Contents

- [ArkNav 🌊🚨](#arknav-)
  - [Table of Contents](#table-of-contents)
  - [1. Overview: What is ArkNav?](#1-overview-what-is-arknav)
    - [Features](#features)
    - [AI Flood Model Summary](#ai-flood-model-summary)
  - [2. Repository Structure](#2-repository-structure)
  - [The repository consists of **two main projects**:](#the-repository-consists-of-two-main-projects)
  - [3. Flask API Server and Flutter Application](#3-flask-api-server-and-flutter-application)
    - [Flask API Server](#flask-api-server)
    - [Flutter Application](#flutter-application)
  - [4. Disclaimers \& Datasets](#4-disclaimers--datasets)
  - [5. Getting Started](#5-getting-started)
    - [Step 1: Clone the Repository](#step-1-clone-the-repository)
    - [Step 2: Run the Flask API Server](#step-2-run-the-flask-api-server)
      - [1. Navigate to the Flask API folder:](#1-navigate-to-the-flask-api-folder)
      - [2. Create a virtual environment and install dependencies:](#2-create-a-virtual-environment-and-install-dependencies)
      - [3. Start the Flask server:](#3-start-the-flask-server)
    - [Step 3: Configure and Run the Flutter App](#step-3-configure-and-run-the-flutter-app)
      - [1. Open the Flutter project:](#1-open-the-flutter-project)
      - [2. Update the baseUrl in the app configuration to match the Flask API server URL:](#2-update-the-baseurl-in-the-app-configuration-to-match-the-flask-api-server-url)
      - [3. Run the Flutter app:](#3-run-the-flutter-app)
    - [6. Enjoy \& Stay Safe! 🌧️🌊](#6-enjoy--stay-safe-️)

---

## 1. Overview: What is ArkNav?

The Philippines lies in the **Pacific Typhoon Belt** and faces flooding from over **20 typhoons every year**. Unfortunately, many existing forecasting systems are **outdated, ineffective, and inefficient** (NDRRMC, 2024; Kim et al., 2023).

**Enter ArkNav** – an **AI-Powered Flood Risk & Knowledge Navigation System** designed to help communities stay safe. ArkNav is both a **mobile and web-based platform** that delivers **local flood forecasts** and **actionable recommendations** to users.

By leveraging datasets from:

- **Project NOAH Flood Dataset**
- **Open-Meteo API** for forecasted weather data
- **Metro Manila Flood Landscape Dataset (Kaggle, by GIOLOGICX)**
- **Philippine Waterway Dataset (NAMRIA)**

...ArkNav uses **machine learning models** like **XGBoost** to provide **accurate 3-day and 7-day flood predictions**.

### Features

- AI flood prediction using **XGBoost** to classify potential flood height based on **19 diverse features**.
- Prescriptive analytics with a **rule-based decision engine** + **LLM** for empathetic, reliable recommendations.
- Optimized evacuation routes using **Google Maps API**.
- Mobile front-end built with **Flutter**; backend using **Flask/Django**.
- Firebase integrations: **real-time notifications**, **multi-location tracking**, **home pinning**, and **AI-certified crowdsourced reporting**.
- Web dashboard for LGUs to monitor **community-level engagement**.
- Iterative development following a **custom Agile methodology** involving **user feedback and LGU collaboration**.

### AI Flood Model Summary

| Property            | Value                                 |
| ------------------- | ------------------------------------- |
| Type                | XGBoost Regressor                     |
| Features            | 19                                    |
| Test Performance    | R² = 0.333, RMSE = 1.57m              |
| Flood Prediction R² | 0.132                                 |
| Train-Test Gap      | 0.040 (stable!)                       |
| Model File          | `models\Arknav_Floodrisk_Modelv2.pkl` |

> ⚠️ **Note:** Limited historical data and paywalled datasets caused some features to be missing. SMART navigation and SMS alerts are limited due to **false positives**, but the model still achieves **R² = 0.333**, acceptable for flood prediction applications.

---

## 2. Repository Structure

## The repository consists of **two main projects**:

```
arknav/
│
├── ArkNav_API/ # Flask API backend project
│ ├── app.py
│ ├── services/
│ ├── models/
│ └── requirements.txt
│
└── ArkNav_Application/ # Flutter mobile application
├── lib/
├── pubspec.yaml
└── assets/
```
---

## 3. Flask API Server and Flutter Application

### Flask API Server
- Handles **data processing, AI flood predictions, and feature engineering**.
- Serves as the **backend** for both web and mobile applications.
- Provides endpoints for **real-time flood forecasts and probability calculations**.

### Flutter Application
- Acts as the **frontend**, providing **mobile users with flood alerts and actionable insights**.
- Offers **interactive maps, evacuation routes, and notifications**.
- Connects to the Flask API server to retrieve **predictions and recommendations**.

> **Summary:** Flask API performs the AI computations, while Flutter delivers a seamless **user experience**.

---

## 4. Disclaimers & Datasets

**Disclaimers:**
- ArkNav predictions are **supplementary** and **should not replace official government warnings**.
- Flood forecasts are based on **available datasets**; real-world conditions may vary.
- Users should **exercise caution** during extreme weather and always follow local authorities.

**Datasets Used:**
- Project NOAH Flood Dataset  
- Open-Meteo API (forecasted weather data)  
- Metro Manila Flood Landscape (Kaggle, by GIOLOGICX)  
- Philippine Waterway Dataset (NAMRIA)  

---

## 5. Getting Started

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/arknav.git
cd arknav
```

### Step 2: Run the Flask API Server
#### 1. Navigate to the Flask API folder:
```cd ArkNav_API```

#### 2. Create a virtual environment and install dependencies:
```
python -m venv .venv
# Activate venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

#### 3. Start the Flask server:
```
python app.py
```
*By default, the server runs at http://127.0.0.1:5000.*

### Step 3: Configure and Run the Flutter App
#### 1. Open the Flutter project:
```
cd ../ArkNav_Application
```
#### 2. Update the baseUrl in the app configuration to match the Flask API server URL:

- For mobile emulator, use http://10.0.2.2:5000 (Android) or your local IP.

- For web, use http://127.0.0.1:5000.

#### 3. Run the Flutter app:
```
flutter run
```
---
### 6. Enjoy & Stay Safe! 🌧️🌊
ArkNav empowers you to stay informed and prepared during floods.

Remember: AI can guide you, but always prioritize official warnings and safety first.


ArkNav – Navigating Floods with AI and Care.

