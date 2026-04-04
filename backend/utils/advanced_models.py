"""
FILE: backend/utils/advanced_models.py
PURPOSE: Machine Learning models for advanced analysis with auto-selection
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class AdvancedPredictor:
    """Advanced prediction engine with multiple ML models"""
    
    def __init__(self):
        self.models = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
    
    def convert_to_numeric(self, series):
        """Convert string series to numeric, removing currency symbols and commas"""
        if isinstance(series, (list, np.ndarray)):
            series = pd.Series(series)
        
        # If already numeric, return as is
        if pd.api.types.is_numeric_dtype(series):
            return series.values
        
        # Convert to string and clean
        cleaned = series.astype(str).str.replace(r'[\$,₹]', '', regex=True)
        cleaned = cleaned.str.replace(',', '', regex=False)
        cleaned = cleaned.str.replace('%', '', regex=False)
        cleaned = cleaned.str.strip()
        
        # Convert to numeric, coercing errors to NaN
        numeric = pd.to_numeric(cleaned, errors='coerce')
        
        # Drop NaN values
        numeric = numeric.dropna()
        
        if len(numeric) == 0:
            raise ValueError("No numeric data found after conversion")
        
        return numeric.values
    
    def find_best_model(self, series):
        """Automatically find the best performing model"""
        try:
            # Convert to numeric
            y = self.convert_to_numeric(series)
            n = len(y)
            
            if n < 10:
                return "linear_regression"
            
            X = np.arange(n).reshape(-1, 1)
            
            # For small datasets, use simple validation
            if n < 30:
                # Use last 20% for testing
                test_size = max(1, int(n * 0.2))
                X_train = X[:-test_size]
                y_train = y[:-test_size]
                X_test = X[-test_size:]
                y_test = y[-test_size:]
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            best_score = -float('inf')
            best_name = "linear_regression"
            
            for name, model in self.models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = r2_score(y_test, y_pred)
                    if score > best_score:
                        best_score = score
                        best_name = name
                except Exception as e:
                    print(f"Model {name} failed: {e}")
                    continue
            
            return best_name
            
        except Exception as e:
            print(f"Auto model selection error: {e}")
            return "linear_regression"
    
    def predict(self, series, model_type="auto", forecast_days=30, column_name="Value", enable_anomaly=False):
        """Run prediction using selected model"""
        
        try:
            # Convert to numeric
            y = self.convert_to_numeric(series)
            n = len(y)
            
            if n < 5:
                return {"error": f"Insufficient data. Only {n} valid numeric values found. Need at least 10."}
            
            # If auto, find best model
            if model_type == "auto":
                model_type = self.find_best_model(y)
                print(f"Auto selected model: {model_type}")
            
            X = np.arange(n).reshape(-1, 1)
            
            # Train model
            model = self._train_model(model_type, X, y)
            
            # Make predictions
            last_idx = len(X)
            future_X = np.arange(last_idx, last_idx + forecast_days).reshape(-1, 1)
            predictions = model.predict(future_X)
            predictions = np.maximum(predictions, 0)  # No negative values
            
            # Calculate metrics (if enough data)
            train_size = max(2, int(n * 0.8))
            if train_size >= 2 and n - train_size >= 1:
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                test_model = self._train_model(model_type, X_train, y_train)
                y_pred_test = test_model.predict(X_test)
                
                mae = float(mean_absolute_error(y_test, y_pred_test))
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
                r2 = float(r2_score(y_test, y_pred_test))
            else:
                mae = 0
                rmse = 0
                r2 = 0.5
            
            # Calculate trend
            last_value = float(y[-1])
            predicted_value = float(predictions[-1])
            
            if predicted_value > last_value:
                trend = "up"
                trend_percent = ((predicted_value - last_value) / last_value) * 100 if last_value != 0 else 0
                recommendation = "📈 BUY / INCREASE"
                trend_icon = "📈"
                trend_text = "UPWARD"
            elif predicted_value < last_value:
                trend = "down"
                trend_percent = ((last_value - predicted_value) / last_value) * 100 if last_value != 0 else 0
                recommendation = "📉 SELL / DECREASE"
                trend_icon = "📉"
                trend_text = "DOWNWARD"
            else:
                trend = "stable"
                trend_percent = 0
                recommendation = "➡️ HOLD / MAINTAIN"
                trend_icon = "➡️"
                trend_text = "STABLE"
            
            # Confidence score
            confidence = min(95, max(50, 50 + (r2 * 40)))
            
            # Detect anomalies
            anomalies = []
            if enable_anomaly:
                anomalies = self._detect_anomalies(y)
            
            # Generate forecast dates
            from datetime import datetime, timedelta
            start_date = datetime.now()
            forecast_dates = [(start_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(forecast_days)]
            
            # Model names for display
            model_display_names = {
                "linear_regression": "Linear Regression",
                "random_forest": "Random Forest",
                "gradient_boosting": "Gradient Boosting"
            }
            
            result = {
                "column": column_name,
                "model_used": model_type,
                "model_display_name": model_display_names.get(model_type, model_type.upper()),
                "current_value": round(last_value, 2),
                "predicted_value": round(predicted_value, 2),
                "predictions": [round(float(p), 2) for p in predictions],
                "forecast_dates": forecast_dates,
                "trend": trend,
                "trend_icon": trend_icon,
                "trend_text": trend_text,
                "trend_percent": round(abs(trend_percent), 2),
                "recommendation": recommendation,
                "confidence": round(confidence, 2),
                "metrics": {
                    "mae": round(mae, 2),
                    "rmse": round(rmse, 2),
                    "r2": round(r2, 4),
                    "data_points": n
                },
                "statistics": {
                    "mean": round(float(np.mean(y)), 2),
                    "median": round(float(np.median(y)), 2),
                    "min": round(float(np.min(y)), 2),
                    "max": round(float(np.max(y)), 2),
                    "std": round(float(np.std(y)), 2)
                },
                "anomalies": anomalies
            }
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _train_model(self, model_type, X, y):
        """Train the selected model"""
        if model_type == "linear_regression":
            model = LinearRegression()
        elif model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
        
        model.fit(X, y)
        return model
    
    def _detect_anomalies(self, series, threshold=2):
        """Detect anomalies using Z-score method"""
        mean = np.mean(series)
        std = np.std(series)
        
        if std == 0:
            return []
        
        z_scores = [(val - mean) / std for val in series]
        anomalies = []
        
        for i, z in enumerate(z_scores):
            if abs(z) > threshold:
                anomalies.append({
                    "index": i,
                    "value": float(series[i]),
                    "z_score": round(z, 2),
                    "severity": "🔴 High" if abs(z) > 3 else "🟡 Medium"
                })
        
        return anomalies
    
    def generate_insights(self, series, prediction_result, column_name):
        """Generate easy-to-understand insights"""
        insights = []
        
        trend = prediction_result["trend"]
        trend_percent = prediction_result["trend_percent"]
        confidence = prediction_result["confidence"]
        current = prediction_result["current_value"]
        predicted = prediction_result["predicted_value"]
        
        # Main trend insight
        if trend == "up":
            insights.append(f"📈 **{column_name} is expected to INCREASE** by {trend_percent:.1f}% over the next {len(prediction_result['predictions'])} periods.")
        elif trend == "down":
            insights.append(f"📉 **{column_name} is expected to DECREASE** by {trend_percent:.1f}% over the next {len(prediction_result['predictions'])} periods.")
        else:
            insights.append(f"➡️ **{column_name} is expected to remain STABLE** over the forecast period.")
        
        # Confidence insight
        if confidence >= 80:
            insights.append(f"✅ **High Confidence ({confidence:.0f}%)** - The model is very confident in this prediction.")
        elif confidence >= 60:
            insights.append(f"📊 **Moderate Confidence ({confidence:.0f}%)** - Consider using with other indicators.")
        else:
            insights.append(f"⚠️ **Low Confidence ({confidence:.0f}%)** - Data may have high volatility. Use caution.")
        
        # Value comparison insight
        if current > prediction_result["statistics"]["mean"]:
            insights.append(f"🎯 Current value ({current:,.2f}) is **ABOVE average** ({prediction_result['statistics']['mean']:,.2f}) - Strong momentum detected.")
        elif current < prediction_result["statistics"]["mean"]:
            insights.append(f"📉 Current value ({current:,.2f}) is **BELOW average** ({prediction_result['statistics']['mean']:,.2f}) - May represent opportunity.")
        
        # Recommendation
        rec = prediction_result["recommendation"]
        if "BUY" in rec:
            insights.append(f"💡 **Recommendation:** {rec} - The upward trend suggests growth potential.")
        elif "SELL" in rec:
            insights.append(f"💡 **Recommendation:** {rec} - The downward trend suggests caution.")
        else:
            insights.append(f"💡 **Recommendation:** {rec} - Wait for clearer signals before acting.")
        
        # Anomaly insight
        if prediction_result.get("anomalies"):
            anomaly_count = len(prediction_result["anomalies"])
            insights.append(f"⚠️ **Detected {anomaly_count} unusual pattern(s)** in the data - investigate these points.")
        
        # Data quality insight
        data_points = prediction_result["metrics"]["data_points"]
        if data_points < 30:
            insights.append(f"📊 **Data note:** Only {data_points} data points available. More data would improve accuracy.")
        
        return insights