import time
import random
import requests
from datetime import datetime, timezone

# Your local FastAPI endpoint
API_URL = "http://localhost:8000/api/readings"

print("Starting IoT Weather Station Simulator...")
print("Press Ctrl+C to stop.\n")

# Initial base values for realistic drift
temp = 24.5
humidity = 60.0
pressure = 1013.25
co2 = 420.0

while True:
    try:
        # Simulate small natural fluctuations
        temp += random.uniform(-0.3, 0.3)
        humidity += random.uniform(-1.0, 1.0)
        pressure += random.uniform(-0.5, 0.5)
        co2 += random.uniform(-2.0, 2.0)

        # Clamp values within realistic boundaries
        temp = max(10.0, min(45.0, temp))
        humidity = max(20.0, min(99.0, humidity))
        pressure = max(970.0, min(1040.0, pressure))
        co2 = max(380.0, min(1200.0, co2))

        # Build payload matching your exact ESP32 JSON schema
        payload = {
            "device_id": "VIRTUAL_ESP32_SIMULATOR",
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "uptime_ms": int(time.time() * 1000),
            "temperature_c": round(temp, 2),
            "humidity_pct": round(humidity, 2),
            "pressure_hpa": round(pressure, 2),
            "light_lux": round(random.uniform(200.0, 8000.0), 2),
            "rain_raw": random.randint(100, 4000),
            "mq135": {
                "adc_raw": random.randint(1200, 2500),
                "adc_pin_voltage_v": round(random.uniform(1.0, 2.5), 2),
                "ao_voltage_v": round(random.uniform(2.0, 4.0), 2),
                "calibrated": True,
                "r0_kohm": 12.5,
                "rs_kohm": round(random.uniform(8.0, 18.0), 2),
                "rs_r0_ratio": round(random.uniform(0.8, 1.5), 2),
                "co2_ppm": round(co2, 2),
                "ammonia_ppm": round(random.uniform(0.1, 1.5), 2),
                "carbon_monoxide_ppm": round(random.uniform(0.05, 0.8), 2),
                "alcohol_ppm": round(random.uniform(0.1, 0.5), 2),
                "toluene_ppm": round(random.uniform(0.01, 0.2), 2),
                "acetone_ppm": round(random.uniform(0.05, 0.4), 2)
            }
        }

        response = requests.post(API_URL, json=payload)
        if response.status_code == 201:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Simulated Reading Sent -> Temp: {temp:.1f}°C, Humidity: {humidity:.1f}%, Pressure: {pressure:.1f}hPa")
        else:
            print(f"Failed to send reading: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Simulator connection error: {e}")

    # Wait 10 seconds before sending the next virtual reading
    time.sleep(10)