import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    select,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
    Session,
)
from sqlalchemy.exc import SQLAlchemyError


# =====================================================
# 1. ENVIRONMENT & CONFIGURATION
# =====================================================
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing from .env")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:5174"
)
origins = [origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()]

# Global holders for trained ML models
rain_classifier_model = None
lstm_temp_model = None
SEQUENCE_LENGTH = 24  # Lookback timesteps required for LSTM


# =====================================================
# 2. PyTorch LSTM MODEL ARCHITECTURE DEFINITION
# =====================================================
class TemperatureLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=1):
        super(TemperatureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take output of last time step
        return out


# =====================================================
# 3. DATABASE SETUP & LIFESPAN
# =====================================================
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

class SensorReadingDB(Base):
    __tablename__ = "sensor_readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    device_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    timestamp_utc: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    received_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    uptime_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    temperature_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pressure_hpa: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    light_lux: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rain_raw: Mapped[int] = mapped_column(Integer, nullable=False)

    mq135_adc_raw: Mapped[int] = mapped_column(Integer, nullable=False)
    mq135_adc_pin_voltage_v: Mapped[float] = mapped_column(Float, nullable=False)
    mq135_ao_voltage_v: Mapped[float] = mapped_column(Float, nullable=False)
    mq135_calibrated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    mq135_r0_kohm: Mapped[float] = mapped_column(Float, nullable=False)
    mq135_rs_kohm: Mapped[float] = mapped_column(Float, nullable=False)
    mq135_rs_r0_ratio: Mapped[float] = mapped_column(Float, nullable=False)

    co2_ppm: Mapped[float] = mapped_column(Float, nullable=False)
    ammonia_ppm: Mapped[float] = mapped_column(Float, nullable=False)
    carbon_monoxide_ppm: Mapped[float] = mapped_column(Float, nullable=False)
    alcohol_ppm: Mapped[float] = mapped_column(Float, nullable=False)
    toluene_ppm: Mapped[float] = mapped_column(Float, nullable=False)
    acetone_ppm: Mapped[float] = mapped_column(Float, nullable=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rain_classifier_model, lstm_temp_model
    print("Initializing database and loading ML models...")
    Base.metadata.create_all(bind=engine)

    # Load Scikit-Learn Rain Classifier
    if os.path.exists("rain_classifier.pkl"):
        try:
            rain_classifier_model = joblib.load("rain_classifier.pkl")
            print("Loaded rain_classifier.pkl successfully.")
        except Exception as e:
            print(f"Warning: Could not load rain_classifier.pkl: {e}")
    else:
        print("Notice: rain_classifier.pkl not found. Classifier will use heuristic fallback.")

    # Load PyTorch LSTM Temperature Model
    if os.path.exists("lstm_temp_model.pt"):
        try:
            lstm_temp_model = TemperatureLSTM(input_dim=3, hidden_dim=64, num_layers=2, output_dim=1)
            lstm_temp_model.load_state_dict(torch.load("lstm_temp_model.pt", map_location=torch.device('cpu')))
            lstm_temp_model.eval()
            print("Loaded lstm_temp_model.pt successfully.")
        except Exception as e:
            print(f"Warning: Could not load lstm_temp_model.pt: {e}")
    else:
        print("Notice: lstm_temp_model.pt not found. LSTM will use heuristic fallback.")

    yield
    print("FastAPI application shutting down.")


# =====================================================
# 4. PYDANTIC SCHEMAS
# =====================================================
class MQ135Reading(BaseModel):
    adc_raw: int
    adc_pin_voltage_v: float
    ao_voltage_v: float
    calibrated: bool
    r0_kohm: float
    rs_kohm: float
    rs_r0_ratio: float
    co2_ppm: float
    ammonia_ppm: float
    carbon_monoxide_ppm: float
    alcohol_ppm: float
    toluene_ppm: float
    acetone_ppm: float

class SensorReading(BaseModel):
    device_id: str = Field(min_length=1, max_length=100)
    timestamp_utc: Optional[str] = None
    uptime_ms: int
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    pressure_hpa: Optional[float] = None
    light_lux: Optional[float] = None
    rain_raw: int
    mq135: MQ135Reading


# =====================================================
# 5. FASTAPI APP & MIDDLEWARE
# =====================================================
app = FastAPI(
    title="ESP32 Environmental Monitoring & ML API",
    description="API for sensor telemetry, sequence history, and automated ML retraining.",
    version="1.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =====================================================
# 6. UTILITY SERIALIZER
# =====================================================
def serialize_reading(reading: SensorReadingDB):
    return {
        "id": reading.id,
        "device_id": reading.device_id,
        "timestamp_utc": reading.timestamp_utc,
        "received_at_utc": reading.received_at_utc.isoformat(),
        "uptime_ms": reading.uptime_ms,
        "temperature_c": reading.temperature_c,
        "humidity_pct": reading.humidity_pct,
        "pressure_hpa": reading.pressure_hpa,
        "light_lux": reading.light_lux,
        "rain_raw": reading.rain_raw,
        "mq135": {
            "adc_raw": reading.mq135_adc_raw,
            "adc_pin_voltage_v": reading.mq135_adc_pin_voltage_v,
            "ao_voltage_v": reading.mq135_ao_voltage_v,
            "calibrated": reading.mq135_calibrated,
            "r0_kohm": reading.mq135_r0_kohm,
            "rs_kohm": reading.mq135_rs_kohm,
            "rs_r0_ratio": reading.mq135_rs_r0_ratio,
            "co2_ppm": reading.co2_ppm,
            "ammonia_ppm": reading.ammonia_ppm,
            "carbon_monoxide_ppm": reading.carbon_monoxide_ppm,
            "alcohol_ppm": reading.alcohol_ppm,
            "toluene_ppm": reading.toluene_ppm,
            "acetone_ppm": reading.acetone_ppm,
        },
    }


# =====================================================
# 7. BACKGROUND MODEL RETRAINING WORKER
# =====================================================
def background_retrain_models():
    """
    Pulls recent rows from PostgreSQL, retrains the Random Forest classifier 
    on the fly, and updates model weights in memory.
    """
    global rain_classifier_model
    db = SessionLocal()
    try:
        statement = select(SensorReadingDB).order_by(SensorReadingDB.received_at_utc.desc()).limit(1000)
        readings = db.execute(statement).scalars().all()
        
        if len(readings) < 30:
            return

        data = [{
            'temperature_c': r.temperature_c,
            'humidity_pct': r.humidity_pct,
            'pressure_hpa': r.pressure_hpa,
            'rain_raw': r.rain_raw,
            'received_at_utc': r.received_at_utc
        } for r in readings if r.temperature_c is not None and r.humidity_pct is not None and r.pressure_hpa is not None]
        
        df = pd.DataFrame(data)
        if df.empty:
            return

        df.sort_values('received_at_utc', inplace=True)
        df['pressure_delta'] = df['pressure_hpa'].diff().fillna(0)
        df['is_raining'] = ((df['humidity_pct'] > 75.0) & (df['pressure_delta'] < -0.2)).astype(int)
        
        X = df[['humidity_pct', 'pressure_delta', 'temperature_c', 'rain_raw']]
        y = df['is_raining']
        
        if len(y.unique()) > 1:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X, y)
            
            joblib.dump(clf, "rain_classifier.pkl")
            rain_classifier_model = clf
            print("Background Task: Rain Classifier successfully auto-retrained with fresh database records.")
    except Exception as e:
        print(f"Error in background retraining worker: {e}")
    finally:
        db.close()


# =====================================================
# 8. CORE ENDPOINTS (Ingest & Read)
# =====================================================
@app.get("/")
def root():
    return {"status": "online", "message": "Weather Station ML API Active", "docs": "/docs"}

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "rain_model_loaded": rain_classifier_model is not None,
            "lstm_model_loaded": lstm_temp_model is not None
        }
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Database unavailable")

@app.post("/api/readings", status_code=201)
def receive_sensor_reading(
    reading: SensorReading, 
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)
):
    mq = reading.mq135
    new_reading = SensorReadingDB(
        device_id=reading.device_id,
        timestamp_utc=reading.timestamp_utc,
        received_at_utc=datetime.now(timezone.utc),
        uptime_ms=reading.uptime_ms,
        temperature_c=reading.temperature_c,
        humidity_pct=reading.humidity_pct,
        pressure_hpa=reading.pressure_hpa,
        light_lux=reading.light_lux,
        rain_raw=reading.rain_raw,
        mq135_adc_raw=mq.adc_raw,
        mq135_adc_pin_voltage_v=mq.adc_pin_voltage_v,
        mq135_ao_voltage_v=mq.ao_voltage_v,
        mq135_calibrated=mq.calibrated,
        mq135_r0_kohm=mq.r0_kohm,
        mq135_rs_kohm=mq.rs_kohm,
        mq135_rs_r0_ratio=mq.ratio if hasattr(mq, 'ratio') else mq.rs_r0_ratio,
        co2_ppm=mq.co2_ppm,
        ammonia_ppm=mq.ammonia_ppm,
        carbon_monoxide_ppm=mq.carbon_monoxide_ppm,
        alcohol_ppm=mq.alcohol_ppm,
        toluene_ppm=mq.toluene_ppm,
        acetone_ppm=mq.acetone_ppm,
    )
    try:
        db.add(new_reading)
        db.commit()
        db.refresh(new_reading)
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to store reading: {e}")

    # Trigger background model retraining asynchronously
    background_tasks.add_task(background_retrain_models)

    return {"status": "success", "reading_id": new_reading.id}

@app.get("/api/readings")
def get_all_readings(
    limit: int = Query(default=100, ge=1, le=500),
    skip: int = Query(default=0, ge=0),
    device_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    statement = select(SensorReadingDB)
    if device_id:
        statement = statement.where(SensorReadingDB.device_id == device_id)
    statement = statement.order_by(SensorReadingDB.received_at_utc.desc()).offset(skip).limit(limit)
    readings = db.execute(statement).scalars().all()
    return {"count": len(readings), "readings": [serialize_reading(r) for r in readings]}

@app.get("/api/readings/latest")
def get_latest_reading(device_id: Optional[str] = None, db: Session = Depends(get_db)):
    statement = select(SensorReadingDB)
    if device_id:
        statement = statement.where(SensorReadingDB.device_id == device_id)
    statement = statement.order_by(SensorReadingDB.received_at_utc.desc()).limit(1)
    reading = db.execute(statement).scalars().first()
    if not reading:
        raise HTTPException(status_code=404, detail="No readings found")
    return serialize_reading(reading)

@app.get("/api/ml/history")
def get_ml_historical_sequence(
    limit: int = Query(default=50, ge=5, le=200),
    device_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    statement = select(SensorReadingDB)
    if device_id:
        statement = statement.where(SensorReadingDB.device_id == device_id)
    statement = statement.order_by(SensorReadingDB.received_at_utc.desc()).limit(limit)
    readings = db.execute(statement).scalars().all()
    readings.reverse()  # Oldest to newest chronological order for LSTM sequence
    return {
        "sequence_length": len(readings),
        "readings": [serialize_reading(r) for r in readings]
    }


# =====================================================
# 9. ACTIVE ML INFERENCE ROUTES
# =====================================================
@app.post("/api/ml/infer/temperature-lstm")
def infer_temperature_lstm(
    horizon_hours: int = Query(default=1, ge=1, le=12, description="Hours ahead to forecast"),
    device_id: Optional[str] = None, 
    db: Session = Depends(get_db)
):
    statement = select(SensorReadingDB).order_by(SensorReadingDB.received_at_utc.desc()).limit(SEQUENCE_LENGTH)
    readings = db.execute(statement).scalars().all()
    
    if len(readings) < 5:
        raise HTTPException(status_code=400, detail="Insufficient historical data for LSTM sequence inference.")

    readings.reverse()  # Chronological order
    latest_temp = readings[-1].temperature_c if readings[-1].temperature_c is not None else 25.0

    if lstm_temp_model is not None:
        try:
            feature_window = [
                [
                    r.temperature_c if r.temperature_c is not None else 0.0,
                    r.humidity_pct if r.humidity_pct is not None else 0.0,
                    r.pressure_hpa if r.pressure_hpa is not None else 1013.0
                ]
                for r in readings
            ]
            
            if len(feature_window) < SEQUENCE_LENGTH:
                padding = [feature_window[0]] * (SEQUENCE_LENGTH - len(feature_window))
                feature_window = padding + feature_window
            else:
                feature_window = feature_window[-SEQUENCE_LENGTH:]

            input_tensor = torch.tensor([feature_window], dtype=torch.float32)
            with torch.no_grad():
                prediction = lstm_temp_model(input_tensor)
                base_forecast = float(prediction.item())
                
            # Scale prediction slightly further out based on requested horizon hours
            # (In a production multi-step LSTM, you'd feed predictions back recursively)
            forecast_temp = base_forecast + ((horizon_hours - 1) * 0.3)
            
        except Exception as e:
            print(f"LSTM Inference error: {e}")
            forecast_temp = latest_temp + (horizon_hours * 0.3)
    else:
        forecast_temp = latest_temp + (horizon_hours * 0.3)

    return {
        "model": "LSTM-Temperature-Forecaster",
        "input_timesteps_used": len(readings),
        "forecast_horizon": f"+{horizon_hours} hour(s)",
        "predicted_temperature_c": round(forecast_temp, 2)
    }


@app.post("/api/ml/infer/rain-classifier")
def infer_rain_classifier(device_id: Optional[str] = None, db: Session = Depends(get_db)):
    statement = select(SensorReadingDB).order_by(SensorReadingDB.received_at_utc.desc()).limit(2)
    readings = db.execute(statement).scalars().all()
    
    if not readings:
        raise HTTPException(status_code=404, detail="No sensor readings available for inference.")

    latest = readings[0]
    pressure_delta = 0.0
    if len(readings) > 1 and latest.pressure_hpa and readings[1].pressure_hpa:
        pressure_delta = latest.pressure_hpa - readings[1].pressure_hpa

    humidity = latest.humidity_pct if latest.humidity_pct is not None else 50.0
    temp = latest.temperature_c if latest.temperature_c is not None else 25.0
    rain_raw = latest.rain_raw

    if rain_classifier_model is not None:
        try:
            features = np.array([[humidity, pressure_delta, temp, rain_raw]])
            prediction = int(rain_classifier_model.predict(features)[0])
            probability = float(rain_classifier_model.predict_proba(features)[0][1])
        except Exception as e:
            print(f"Classifier Inference error: {e}")
            prediction = 1 if humidity > 75.0 else 0
            probability = 0.85 if prediction == 1 else 0.15
    else:
        prediction = 1 if (humidity > 75.0 and pressure_delta < -0.3) else 0
        probability = 0.88 if prediction == 1 else 0.12

    return {
        "model": "RandomForest-Rain-Classifier",
        "features_evaluated": {
            "humidity_pct": humidity,
            "pressure_delta_hpa": round(pressure_delta, 2),
            "rain_sensor_adc": rain_raw
        },
        "will_rain": bool(prediction),
        "rain_probability": round(probability, 2)
    }