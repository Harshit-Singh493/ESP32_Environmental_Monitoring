# 🌦️ IoT-Based Weather Station with Real-Time Neural Forecasting & Automated ML Retraining

A full-stack, production-grade Internet of Things (IoT) environmental monitoring system coupled with a Machine Learning (ML) pipeline. Built using an **ESP32 microcontroller**, a **FastAPI PostgreSQL backend**, an **automated background retraining worker**, and a **React + Tailwind CSS dashboard**.

---

## 🏗️ System Architecture & Workflow

```text
[ ESP32 Sensor Hardware ] 
  ├── BME280 (Temp, Humidity, Pressure)
  ├── BH1750 (Ambient Light / Lux)
  ├── MQ135  (Multi-Gas / CO2 Estimate)
  └── Rain Sensor (Analog Precipitation Detection)
          │
          │  (HTTP POST / JSON Payload via Wi-Fi / mDNS)
          ▼
[ FastAPI Backend (Python) ] ──> [ PostgreSQL Database ]
          │                               │
          ├─► Background Worker           │ (Historical Time-Series)
          │     └─► Auto-Retrains ML Model  │
          │                               ▼
          └─► ML Inference Endpoints ◄────┘
                ├── PyTorch LSTM (Temperature Forecasting)
                └── Random Forest (Rain Classification)
                          │
                          ▼
             [ React Dashboard (Vite + Recharts) ]
```

## 📂 Project Directory Structure
```text
📁 ESP32_Environmental_Monitoring/
│
├── 📁 backend/                  # FastAPI Backend & ML Pipeline
│   ├── 📄 main.py               # REST API, DB endpoints, and ML inference routes
│   ├── 📄 train_classifier.py   # Random Forest Rain Classifier training script
│   ├── 📄 train_lstm.py         # PyTorch LSTM Temperature Forecaster training script
│   ├── 📄 export_data.py        # Utility script to export PostgreSQL rows to CSV
│   ├── 📄 simulator.py          # Virtual ESP32 simulator for testing without hardware
│   ├── 📄 .env                  # Environment configurations (Database URL, CORS)
│   ├── 📄 requirements.txt      # Python dependencies
│   └── 📁 venv/                 # Python virtual environment
│
├── 📁 frontend/                 # React + Vite Dashboard
│   ├── 📁 src/
│   │   ├── 📄 App.jsx           # Main dashboard component (Charts, Cards, Horizon Selector)
│   │   ├── 📄 main.jsx
│   │   └── 📄 index.css         # Tailwind CSS styles
│   ├── 📄 package.json          # Node dependencies (Recharts, Lucide-react)
│   └── ...
│
├── 📁 sketch/                   # Arduino Firmware
│   ├── 📄 sketch_oct29a.ino     # ESP32 firmware code (mDNS, NVS Preferences, HTTP POST)
│   └── 📄 config.example.h      # Template configuration file for Wi-Fi & server credentials
│
└── 📄 README.md                 # Project documentation
```