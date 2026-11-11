import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area 
} from 'recharts';
import { 
  Thermometer, Droplets, Gauge, Sun, CloudRain, Wind, Activity, RefreshCw, Cpu 
} from 'lucide-react';

const API_BASE_URL = "http://localhost:8000";

export default function WeatherDashboard() {
  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState([]);
  const [mlForecast, setMlForecast] = useState(null);
  const [mlRain, setMlRain] = useState(null);
  const [forecastHorizon, setForecastHorizon] = useState(1); // Admin forecast horizon state
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch latest reading and historical sequence
  const fetchData = async () => {
    try {
      const [latestRes, historyRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/readings/latest`),
        fetch(`${API_BASE_URL}/api/ml/history?limit=30`)
      ]);

      if (!latestRes.ok || !historyRes.ok) throw new Error("Failed to fetch from backend");

      const latestData = await latestRes.json();
      const historyData = await historyRes.json();

      setLatest(latestData);
      
      const formattedHistory = historyData.readings.map(r => ({
        ...r,
        timeLabel: new Date(r.received_at_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      }));
      
      setHistory(formattedHistory);
      setError(null);
    } catch (err) {
      setError("Cannot connect to FastAPI backend. Ensure server is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  // Run ML Inference with dynamic horizon
  const runMLInference = async (horizon = forecastHorizon) => {
    try {
      const [tempRes, rainRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/ml/infer/temperature-lstm?horizon_hours=${horizon}`, { method: 'POST' }),
        fetch(`${API_BASE_URL}/api/ml/infer/rain-classifier`, { method: 'POST' })
      ]);
      
      if (tempRes.ok) setMlForecast(await tempRes.json());
      if (rainRes.ok) setMlRain(await rainRes.json());
    } catch (err) {
      console.error("ML Inference error:", err);
    }
  };

  // Handle horizon dropdown change
  const handleHorizonChange = (e) => {
    const newHorizon = Number(e.target.value);
    setForecastHorizon(newHorizon);
    runMLInference(newHorizon);
  };

  useEffect(() => {
    fetchData();
    runMLInference(forecastHorizon);

    // Poll live telemetry and predictions every 10 seconds
    const interval = setInterval(() => {
      fetchData();
      runMLInference(forecastHorizon);
    }, 10000);

    return () => clearInterval(interval);
  }, [forecastHorizon]);

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-neutral-950 text-neutral-300">
        <RefreshCw className="animate-spin h-5 w-5 text-neutral-400 mr-3" />
        <span className="text-sm font-medium tracking-wide">Loading Weather Station...</span>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 p-8 font-sans antialiased">
      
      {/* HEADER */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 border-b border-neutral-800 pb-5">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-neutral-100">
            IoT Weather Station & ML Analysis
          </h1>
          <p className="text-neutral-400 text-sm mt-1">
            Device ID: <span className="text-neutral-200 font-mono">{latest?.device_id || "ESP32_WS_01"}</span> | Real-time Sensor Telemetry & Neural Forecasting
          </p>
        </div>
        <div className="flex items-center gap-3 mt-4 md:mt-0">
          <span className="inline-flex items-center px-3 py-1 rounded text-xs font-medium bg-neutral-900 text-neutral-300 border border-neutral-800">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mr-2"></span>
            System Online
          </span>
          <button 
            onClick={() => { fetchData(); runMLInference(forecastHorizon); }}
            className="flex items-center gap-2 bg-neutral-900 hover:bg-neutral-800 text-neutral-300 border border-neutral-800 px-3.5 py-1.5 rounded text-xs font-medium transition"
          >
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </button>
        </div>
      </header>

      {error && (
        <div className="mb-6 bg-neutral-900 border border-rose-900/50 text-rose-300 px-4 py-3 rounded text-sm">
          {error}
        </div>
      )}

      {/* METRIC CARDS GRID */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        
        {/* Temperature */}
        <div className="bg-neutral-900 border border-neutral-800 p-5 rounded-lg shadow-sm">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-xs font-medium uppercase tracking-wider text-neutral-400">Temperature</p>
              <h3 className="text-2xl font-semibold mt-2 text-neutral-100">{latest?.temperature_c ?? "--"} <span className="text-sm font-normal text-neutral-400">°C</span></h3>
            </div>
            <div className="p-2 bg-neutral-800 text-neutral-300 rounded">
              <Thermometer className="h-5 w-5" />
            </div>
          </div>
          <p className="text-xs text-neutral-500 mt-3">BME280 Sensor</p>
        </div>

        {/* Humidity */}
        <div className="bg-neutral-900 border border-neutral-800 p-5 rounded-lg shadow-sm">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-xs font-medium uppercase tracking-wider text-neutral-400">Humidity</p>
              <h3 className="text-2xl font-semibold mt-2 text-neutral-100">{latest?.humidity_pct ?? "--"} <span className="text-sm font-normal text-neutral-400">%</span></h3>
            </div>
            <div className="p-2 bg-neutral-800 text-neutral-300 rounded">
              <Droplets className="h-5 w-5" />
            </div>
          </div>
          <p className="text-xs text-neutral-500 mt-3">Relative Air Saturation</p>
        </div>

        {/* Barometric Pressure */}
        <div className="bg-neutral-900 border border-neutral-800 p-5 rounded-lg shadow-sm">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-xs font-medium uppercase tracking-wider text-neutral-400">Pressure</p>
              <h3 className="text-2xl font-semibold mt-2 text-neutral-100">{latest?.pressure_hpa ?? "--"} <span className="text-sm font-normal text-neutral-400">hPa</span></h3>
            </div>
            <div className="p-2 bg-neutral-800 text-neutral-300 rounded">
              <Gauge className="h-5 w-5" />
            </div>
          </div>
          <p className="text-xs text-neutral-500 mt-3">Sea-level adjusted</p>
        </div>

        {/* Air Quality (CO2) */}
        <div className="bg-neutral-900 border border-neutral-800 p-5 rounded-lg shadow-sm">
          <div className="flex justify-between items-start">
            <div>
              <p className="text-xs font-medium uppercase tracking-wider text-neutral-400">CO2 Estimate</p>
              <h3 className="text-2xl font-semibold mt-2 text-neutral-100">{latest?.mq135?.co2_ppm?.toFixed(0) ?? "--"} <span className="text-sm font-normal text-neutral-400">ppm</span></h3>
            </div>
            <div className="p-2 bg-neutral-800 text-neutral-300 rounded">
              <Wind className="h-5 w-5" />
            </div>
          </div>
          <p className="text-xs text-neutral-500 mt-3">MQ135 Gas Sensor</p>
        </div>

      </div>

      {/* CHARTS SECTION */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        
        {/* Temperature Trend */}
        <div className="bg-neutral-900 border border-neutral-800 p-5 rounded-lg shadow-sm">
          <h3 className="text-sm font-semibold mb-4 flex items-center gap-2 text-neutral-200">
            <Thermometer className="h-4 w-4 text-neutral-400" /> Temperature Trend (°C)
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={history}>
                <defs>
                  <linearGradient id="tempColor" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#a3a3a3" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="#a3a3a3" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" opacity={0.6} />
                <XAxis dataKey="timeLabel" stroke="#737373" fontSize={11} />
                <YAxis stroke="#737373" fontSize={11} domain={['auto', 'auto']} />
                <Tooltip contentStyle={{ backgroundColor: '#171717', borderColor: '#404040', borderRadius: '4px', color: '#e5e5e5' }} />
                <Area type="monotone" dataKey="temperature_c" stroke="#d4d4d4" strokeWidth={1.5} fillOpacity={1} fill="url(#tempColor)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Humidity Trend */}
        <div className="bg-neutral-900 border border-neutral-800 p-5 rounded-lg shadow-sm">
          <h3 className="text-sm font-semibold mb-4 flex items-center gap-2 text-neutral-200">
            <Droplets className="h-4 w-4 text-neutral-400" /> Humidity Trend (%)
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" opacity={0.6} />
                <XAxis dataKey="timeLabel" stroke="#737373" fontSize={11} />
                <YAxis stroke="#737373" fontSize={11} domain={[0, 100]} />
                <Tooltip contentStyle={{ backgroundColor: '#171717', borderColor: '#404040', borderRadius: '4px', color: '#e5e5e5' }} />
                <Line type="monotone" dataKey="humidity_pct" stroke="#a3a3a3" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      {/* ML INFERENCE SECTION */}
      <div className="bg-neutral-900 border border-neutral-800 p-5 rounded-lg shadow-sm">
        <div className="flex justify-between items-center mb-5">
          <h3 className="text-sm font-semibold flex items-center gap-2 text-neutral-200">
            <Cpu className="h-4 w-4 text-neutral-400" /> ML Model Predictions & Analysis
          </h3>
          <button 
            onClick={() => runMLInference(forecastHorizon)}
            className="bg-neutral-800 hover:bg-neutral-700 text-neutral-200 border border-neutral-700 px-3 py-1.5 rounded text-xs font-medium transition"
          >
            Run Inference
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          
          {/* LSTM Forecast Card */}
          <div className="bg-neutral-950 border border-neutral-800 p-4 rounded">
            <div className="flex justify-between items-center mb-2">
              <span className="text-[11px] font-medium uppercase tracking-wider text-neutral-400 bg-neutral-900 px-2 py-0.5 rounded border border-neutral-800">
                LSTM Time-Series Regressor
              </span>
              
              {/* ADMIN FORECAST HORIZON SELECTOR */}
              <div className="flex items-center gap-1.5 text-xs text-neutral-400">
                <span>Horizon:</span>
                <select 
                  value={forecastHorizon} 
                  onChange={handleHorizonChange}
                  className="bg-neutral-900 border border-neutral-700 text-neutral-200 rounded px-2 py-1 text-xs focus:outline-none"
                >
                  <option value={1}>+1 Hr</option>
                  <option value={2}>+2 Hr</option>
                  <option value={3}>+3 Hr</option>
                  <option value={4}>+4 Hr</option>
                  <option value={6}>+6 Hr</option>
                </select>
              </div>
            </div>

            <div className="mt-3 flex justify-between items-baseline">
              <div>
                <p className="text-xs text-neutral-400">Predicted Temperature ({mlForecast?.forecast_horizon || `+${forecastHorizon} Hr`}):</p>
                <h4 className="text-xl font-semibold text-neutral-100 mt-0.5">
                  {mlForecast ? `${mlForecast.predicted_temperature_c} °C` : "Loading..."}
                </h4>
              </div>
              <div className="text-right">
                <p className="text-[11px] text-neutral-500">Lookback window</p>
                <p className="text-xs font-mono text-neutral-300">{mlForecast?.input_timesteps_used || 24} steps</p>
              </div>
            </div>
            <p className="text-xs text-neutral-400 mt-3 pt-2.5 border-t border-neutral-900">
              Captures non-linear thermal cycles compared against linear ridge regression baseline.
            </p>
          </div>

          {/* Rain Classifier Card */}
          <div className="bg-neutral-950 border border-neutral-800 p-4 rounded">
            <span className="text-[11px] font-medium uppercase tracking-wider text-neutral-400 bg-neutral-900 px-2 py-0.5 rounded border border-neutral-800">
              Random Forest Classifier
            </span>
            <div className="mt-3 flex justify-between items-baseline">
              <div>
                <p className="text-xs text-neutral-400">Rain Probability:</p>
                <h4 className="text-xl font-semibold text-neutral-100 mt-0.5">
                  {mlRain ? `${(mlRain.rain_probability * 100).toFixed(1)}%` : "Loading..."}
                </h4>
              </div>
              <div className="text-right">
                <p className="text-[11px] text-neutral-500">Status</p>
                <span className={`inline-block px-2 py-0.5 rounded text-[11px] font-medium mt-0.5 ${mlRain?.will_rain ? 'bg-neutral-800 text-neutral-200 border border-neutral-700' : 'bg-neutral-900 text-neutral-400 border border-neutral-800'}`}>
                  {mlRain?.will_rain ? 'Rain Expected' : 'Dry Conditions'}
                </span>
              </div>
            </div>
            <p className="text-xs text-neutral-400 mt-3 pt-2.5 border-t border-neutral-900">
              Evaluates barometric pressure tendency (ΔP) and relative humidity. Auto-retrains on new data.
            </p>
          </div>

        </div>
      </div>

    </div>
  );
}