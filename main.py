"""
FastAPI Backend for Retail Sales Prediction
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = FastAPI(title="Retail Sales ML API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SalesData(BaseModel):
    date: str
    sales: float

class PredictionResponse(BaseModel):
    models: List[dict]
    best_model: str
    test_predictions: List[dict]
    future_predictions: List[dict]
    feature_importance: List[dict]

class MLPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_features(self, df):
        """Engineer features"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Time features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Cyclical features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'sales_rolling_mean_{window}'] = df['sales'].shift(1).rolling(window=window).mean()
            df[f'sales_rolling_std_{window}'] = df['sales'].shift(1).rolling(window=window).std()
        
        # Drop NaN
        df = df.dropna()
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for training"""
        feature_cols = [col for col in df.columns if col not in ['date', 'sales']]
        self.feature_names = feature_cols
        
        X = df[feature_cols]
        y = df['sales']
        
        # Split chronologically
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, df.iloc[split_idx:]['date']
    
    def train_and_predict(self, df):
        """Train models and make predictions"""
        # Create features
        df_featured = self.create_features(df)
        
        # Prepare data
        feature_cols = [col for col in df_featured.columns if col not in ['date', 'sales']]
        self.feature_names = feature_cols
        
        X = df_featured[feature_cols]
        y = df_featured['sales']
        dates = df_featured['date']
        
        # Split chronologically
        split_idx = int(len(df_featured) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        test_dates = dates[split_idx:]
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        models_dict = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        }
        
        results = []
        test_predictions = None
        best_model_name = None
        best_mae = float('inf')
        
        for name, model in models_dict.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            results.append({
                'name': name,
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape)
            })
            
            self.models[name] = model
            
            if mae < best_mae:
                best_mae = mae
                best_model_name = name
                test_predictions = [
                    {
                        'date': date.strftime('%Y-%m-%d'),
                        'actual': float(actual),
                        'predicted': float(pred)
                    }
                    for date, actual, pred in zip(test_dates, y_test, y_pred)
                ]
        
        # Feature importance
        best_model = self.models[best_model_name]
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            max_importance = max(importances) if max(importances) > 0 else 1
            feature_importance = [
                {
                    'feature': feat, 
                    'importance': float(imp / max_importance * 100)
                }
                for feat, imp in zip(self.feature_names, importances)
            ]
            feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:10]
        else:
            # For Linear Regression, use coefficient magnitudes
            if hasattr(best_model, 'coef_'):
                coefs = np.abs(best_model.coef_)
                max_coef = max(coefs) if max(coefs) > 0 else 1
                feature_importance = [
                    {
                        'feature': feat,
                        'importance': float(abs(coef) / max_coef * 100)
                    }
                    for feat, coef in zip(self.feature_names, coefs)
                ]
                feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:10]
            else:
                feature_importance = []
        
        # Future predictions with proper feature updates
        last_date = df['date'].max()
        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)
        
        future_predictions = []
        recent_sales = df['sales'].tail(30).values.tolist()
        
        for i in range(1, 31):
            future_date = last_date + timedelta(days=i)
            
            # Time features for this future date
            day_of_week = future_date.weekday()
            month = future_date.month
            day_of_month = future_date.day
            week_of_year = future_date.isocalendar().week
            is_weekend = 1 if day_of_week >= 5 else 0
            quarter = (month - 1) // 3 + 1
            day_of_year = future_date.timetuple().tm_yday
            
            # Cyclical features
            day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            
            # Holiday detection
            is_holiday = 0
            if month == 12 and day_of_month > 15:
                is_holiday = 1
            elif month == 11 and day_of_month > 20:
                is_holiday = 1
            
            # Lag features
            sales_lag_1 = recent_sales[-1] if len(recent_sales) >= 1 else df['sales'].mean()
            sales_lag_7 = recent_sales[-7] if len(recent_sales) >= 7 else df['sales'].mean()
            sales_lag_14 = recent_sales[-14] if len(recent_sales) >= 14 else df['sales'].mean()
            sales_lag_30 = recent_sales[-30] if len(recent_sales) >= 30 else df['sales'].mean()
            
            # Rolling features
            sales_rolling_mean_7 = np.mean(recent_sales[-7:]) if len(recent_sales) >= 7 else df['sales'].mean()
            sales_rolling_std_7 = np.std(recent_sales[-7:]) if len(recent_sales) >= 7 else df['sales'].std()
            sales_rolling_mean_14 = np.mean(recent_sales[-14:]) if len(recent_sales) >= 14 else df['sales'].mean()
            sales_rolling_std_14 = np.std(recent_sales[-14:]) if len(recent_sales) >= 14 else df['sales'].std()
            sales_rolling_mean_30 = np.mean(recent_sales[-30:]) if len(recent_sales) >= 30 else df['sales'].mean()
            sales_rolling_std_30 = np.std(recent_sales[-30:]) if len(recent_sales) >= 30 else df['sales'].std()
            
            # Build feature dict
            feature_dict = {
                'day_of_week': day_of_week,
                'month': month,
                'day_of_month': day_of_month,
                'week_of_year': week_of_year,
                'is_weekend': is_weekend,
                'is_holiday': is_holiday,
                'quarter': quarter,
                'day_of_year': day_of_year,
                'day_of_week_sin': day_of_week_sin,
                'day_of_week_cos': day_of_week_cos,
                'month_sin': month_sin,
                'month_cos': month_cos,
                'sales_lag_1': sales_lag_1,
                'sales_lag_7': sales_lag_7,
                'sales_lag_14': sales_lag_14,
                'sales_lag_30': sales_lag_30,
                'sales_rolling_mean_7': sales_rolling_mean_7,
                'sales_rolling_std_7': sales_rolling_std_7,
                'sales_rolling_mean_14': sales_rolling_mean_14,
                'sales_rolling_std_14': sales_rolling_std_14,
                'sales_rolling_mean_30': sales_rolling_mean_30,
                'sales_rolling_std_30': sales_rolling_std_30
            }
            
            # Create feature array in correct order
            X_future = np.array([[feature_dict.get(feat, 0) for feat in feature_cols]])
            X_future_scaled = self.scaler.transform(X_future)
            
            # Predict
            pred = best_model.predict(X_future_scaled)[0]
            pred = max(pred, 0)
            
            # Update recent sales
            recent_sales.append(pred)
            if len(recent_sales) > 30:
                recent_sales = recent_sales[-30:]
            
            future_predictions.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'predicted': float(pred)
            })
        
        return {
            'models': results,
            'best_model': best_model_name,
            'test_predictions': test_predictions,
            'future_predictions': future_predictions,
            'feature_importance': feature_importance
        }

# Global predictor
predictor = MLPredictor()

@app.get("/")
async def root():
    return {
        "message": "Retail Sales ML API",
        "version": "1.0",
        "endpoints": {
            "POST /upload": "Upload CSV and get predictions",
            "POST /predict": "Send JSON data and get predictions",
            "GET /sample": "Get sample data",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/sample")
async def get_sample_data():
    """Generate sample data"""
    start_date = datetime(2023, 1, 1)
    data = []
    
    for i in range(365):
        date = start_date + timedelta(days=i)
        day_of_week = date.weekday()
        month = date.month
        
        sales = 1000
        sales += np.sin(i / 365 * 2 * np.pi) * 200
        sales += 300 if day_of_week in [5, 6] else 0
        sales += 500 if month == 12 else 0
        sales += np.random.normal(0, 50)
        sales += i * 0.5
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'sales': round(sales, 2)
        })
    
    return {"data": data}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV file and get predictions"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if 'date' not in df.columns or 'sales' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must have 'date' and 'sales' columns")
        
        df['date'] = pd.to_datetime(df['date'])
        df['sales'] = df['sales'].astype(float)
        
        results = predictor.train_and_predict(df)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(data: List[SalesData]):
    """Send JSON data and get predictions"""
    try:
        df = pd.DataFrame([{'date': item.date, 'sales': item.sales} for item in data])
        
        if len(df) < 100:
            raise HTTPException(status_code=400, detail="Need at least 100 data points")
        
        results = predictor.train_and_predict(df)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
