import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("BANANA PRODUCTION FORECASTING AI - TWO-PHASE APPROACH")
print("=" * 70)
print("\nPhase 1: Climate Prediction")
print("  - Use simple regression to forecast future climate variables")
print("  - Project temperature, humidity, GDD, and climate range")
print("\nPhase 2: Production Prediction")
print("  - Train neural network on historical climate-production relationships")
print("  - Use predicted climate data to forecast banana production")
print("=" * 70)

# ============================================
# PHASE 1: CLIMATE PREDICTION MODELS
# ============================================

class ClimatePredictor:
    """
    Uses linear regression to predict future climate trends
    More reliable for small datasets than complex models
    """
    def __init__(self):
        self.models = {}
        self.scalers = {}
    
    def fit(self, df, climate_cols, year_col='year'):
        """Train linear models for each climate variable"""
        years = df[year_col].values.reshape(-1, 1)
        
        for col in climate_cols:
            values = df[col].values
            model = LinearRegression()
            model.fit(years, values)
            self.models[col] = model
            
            # Store statistics for adding realistic variation
            residuals = values - model.predict(years)
            self.scalers[col] = np.std(residuals)
    
    def predict(self, future_years, add_variation=True):
        """Predict future climate with optional natural variation"""
        future_years_array = np.array(future_years).reshape(-1, 1)
        predictions = {}
        
        for col, model in self.models.items():
            base_pred = model.predict(future_years_array)
            
            if add_variation:
                # Add realistic year-to-year variation
                noise = np.random.normal(0, self.scalers[col] * 0.8, len(future_years))
                predictions[col] = base_pred + noise
            else:
                predictions[col] = base_pred
        
        return predictions

# ============================================
# PHASE 2: PRODUCTION PREDICTION MODEL
# ============================================

class ProductionNet(nn.Module):
    """
    Small neural network optimized for limited data
    Simpler architecture prevents overfitting
    """
    def __init__(self, input_size):
        super(ProductionNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),  # Regularization for small datasets
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def train_production_model(X, y, input_size, epochs=1500):
    """
    Train production model with techniques suitable for small datasets
    """
    model = ProductionNet(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    # Training with early stopping awareness
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        # Track improvement
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    return model

# ============================================
# DATA LOADING
# ============================================

def load_data():
    """Load both datasets"""
    print("\nLoading datasets...")
    
    # Load regional data
    regional_df = pd.read_csv('Brazil-All(By-Region).csv')
    regional_df = regional_df.sort_values(['region', 'year']).reset_index(drop=True)
    
    # Load aggregated data
    aggregated_df = pd.read_csv('Brazil-Total(Aggregate-aka-All-Brazil.csv')
    aggregated_df = aggregated_df.sort_values('year').reset_index(drop=True)
    
    # Clean data
    regional_df = regional_df.dropna()
    aggregated_df = aggregated_df.dropna()
    
    print(f"Regional data: {len(regional_df)} records across {regional_df['region'].nunique()} regions")
    print(f"National data: {len(aggregated_df)} records")
    print(f"Year range: {aggregated_df['year'].min()} - {aggregated_df['year'].max()}")
    
    return regional_df, aggregated_df

# ============================================
# NATIONAL FORECASTING PIPELINE
# ============================================

def forecast_national(aggregated_df, future_years):
    """
    Complete pipeline for national-level forecasting
    Phase 1: Predict climate
    Phase 2: Predict production from climate
    """
    print("\n" + "=" * 70)
    print("NATIONAL FORECASTING")
    print("=" * 70)
    
    # Phase 1: Train climate prediction models
    print("\nPHASE 1: Climate Prediction")
    print("-" * 40)
    
    climate_cols = ['total_mean', 'total_max', 'total_min', 'total_humidity', 
                    'total_range', 'total_GDD']
    
    climate_predictor = ClimatePredictor()
    climate_predictor.fit(aggregated_df, climate_cols, 'year')
    
    # Predict future climate
    future_climate = climate_predictor.predict(future_years, add_variation=True)
    
    print("Climate trends calculated:")
    for col in climate_cols[:3]:  # Show sample
        model = climate_predictor.models[col]
        slope = model.coef_[0]
        print(f"  {col}: {slope:+.4f} units/year")
    
    # Predict future area using simple regression
    area_model = LinearRegression()
    years = aggregated_df['year'].values.reshape(-1, 1)
    areas = aggregated_df['total_areaKM'].values
    area_model.fit(years, areas)
    future_areas = area_model.predict(np.array(future_years).reshape(-1, 1))
    future_areas = np.maximum(future_areas, 0)  # Ensure positive
    
    # Phase 2: Train production prediction model
    print("\nPHASE 2: Production Prediction")
    print("-" * 40)
    print("Training neural network on climate-production relationships...")
    
    # Prepare training data
    feature_cols = ['total_areaKM', 'total_mean', 'total_max', 'total_min', 
                    'total_humidity', 'total_range', 'total_GDD']
    
    X = aggregated_df[feature_cols].values
    y = aggregated_df['total_prod'].values
    
    # Standardize features (better for small datasets than MinMax)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Train model
    model = train_production_model(X_scaled, y_scaled, len(feature_cols), epochs=1500)
    
    # Phase 3: Generate production forecasts
    print("\nGenerating production forecasts from predicted climate...")
    
    # Create future feature matrix
    future_X = np.column_stack([
        future_areas,
        future_climate['total_mean'],
        future_climate['total_max'],
        future_climate['total_min'],
        future_climate['total_humidity'],
        future_climate['total_range'],
        future_climate['total_GDD']
    ])
    
    future_X_scaled = scaler_X.transform(future_X)
    
    # Predict production
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(future_X_scaled)
        predictions_scaled = model(X_tensor).numpy()
        predictions = scaler_y.inverse_transform(predictions_scaled)
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'year': future_years,
        'total_prod': predictions.flatten(),
        'total_areaKM': future_areas,
        'total_mean': future_climate['total_mean'],
        'total_max': future_climate['total_max'],
        'total_min': future_climate['total_min'],
        'total_humidity': future_climate['total_humidity'],
        'total_GDD': future_climate['total_GDD'],
        'total_range': future_climate['total_range']
    })
    
    predictions_df['total_prodperareaKM'] = (
        predictions_df['total_prod'] / predictions_df['total_areaKM']
    )
    
    print(f"Forecasted production for {len(future_years)} years")
    
    return predictions_df

# ============================================
# REGIONAL FORECASTING PIPELINE
# ============================================

def forecast_regional(regional_df, future_years):
    """
    Complete pipeline for regional forecasting
    Phase 1: Predict climate per region
    Phase 2: Predict production per region from climate
    """
    print("\n" + "=" * 70)
    print("REGIONAL FORECASTING")
    print("=" * 70)
    
    regions = regional_df['region'].unique()
    all_predictions = []
    
    climate_cols = ['climate_Mean', 'climate_Max', 'climate_Min', 
                    'climate_Humidity', 'climate_Range', 'climate_GDD']
    feature_cols = ['areaKM', 'climate_Mean', 'climate_Max', 'climate_Min',
                    'climate_Humidity', 'climate_Range', 'climate_GDD']
    
    for region in regions:
        print(f"\n{region}")
        print("-" * 40)
        
        region_data = regional_df[regional_df['region'] == region].copy()
        
        if len(region_data) < 5:
            print(f"Skipping - insufficient data (n={len(region_data)})")
            continue
        
        # Phase 1: Predict climate for this region
        print("Phase 1: Predicting regional climate...")
        climate_predictor = ClimatePredictor()
        climate_predictor.fit(region_data, climate_cols, 'year')
        future_climate = climate_predictor.predict(future_years, add_variation=True)
        
        # Predict area
        area_model = LinearRegression()
        years = region_data['year'].values.reshape(-1, 1)
        areas = region_data['areaKM'].values
        area_model.fit(years, areas)
        future_areas = area_model.predict(np.array(future_years).reshape(-1, 1))
        future_areas = np.maximum(future_areas, 0)
        
        # Phase 2: Train production model
        print("Phase 2: Training production model...")
        X = region_data[feature_cols].values
        y = region_data['prod'].values
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        model = train_production_model(X_scaled, y_scaled, len(feature_cols), epochs=1500)
        
        # Generate predictions
        future_X = np.column_stack([
            future_areas,
            future_climate['climate_Mean'],
            future_climate['climate_Max'],
            future_climate['climate_Min'],
            future_climate['climate_Humidity'],
            future_climate['climate_Range'],
            future_climate['climate_GDD']
        ])
        
        future_X_scaled = scaler_X.transform(future_X)
        
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(future_X_scaled)
            predictions_scaled = model(X_tensor).numpy()
            predictions = scaler_y.inverse_transform(predictions_scaled)
        
        # Store predictions
        for i, year in enumerate(future_years):
            all_predictions.append({
                'id': f"{region}_{year}",
                'region': region,
                'year': year,
                'areaKM': future_areas[i],
                'prod': predictions[i][0],
                'climate_Mean': future_climate['climate_Mean'][i],
                'climate_Max': future_climate['climate_Max'][i],
                'climate_Min': future_climate['climate_Min'][i],
                'climate_Humidity': future_climate['climate_Humidity'][i],
                'climate_Range': future_climate['climate_Range'][i],
                'climate_GDD': future_climate['climate_GDD'][i]
            })
        
        print(f"Forecasted {len(future_years)} years")
    
    return pd.DataFrame(all_predictions)

# ============================================
# VISUALIZATION AND EXPORT
# ============================================

def save_results(historical_df, predictions_df, level='national'):
    """Save combined results and create visualization"""
    
    if level == 'national':
        filename_csv = 'brazil_future_predictions.csv'
        filename_png = 'brazil_forecast.png'
        title = 'Brazil Banana Production: Historical and Forecast'
        ylabel = 'Total Production (metric tons)'
        
        combined = pd.concat([historical_df, predictions_df], ignore_index=True)
        combined = combined.sort_values('year').reset_index(drop=True)
        combined.to_csv(filename_csv, index=False)
        
        plt.figure(figsize=(12, 6))
        plt.plot(combined['year'], combined['total_prod'], 
                'o-', linewidth=2, markersize=5, color='steelblue', alpha=0.8)
        
        last_year = historical_df['year'].max()
        plt.axvline(x=last_year, color='gray', linestyle='--', 
                   linewidth=1.5, alpha=0.5, label='Forecast Start')
        
    else:  # regional
        filename_csv = 'regional_future_predictions.csv'
        filename_png = 'regional_forecast.png'
        title = 'Regional Banana Production: Historical and Forecast'
        ylabel = 'Production (metric tons)'
        
        combined = pd.concat([historical_df, predictions_df], ignore_index=True)
        combined = combined.sort_values(['region', 'year']).reset_index(drop=True)
        combined.to_csv(filename_csv, index=False)
        
        regions = combined['region'].unique()
        cmap = plt.colormaps.get_cmap('tab10')
        
        plt.figure(figsize=(14, 7))
        
        for idx, region in enumerate(regions):
            region_data = combined[combined['region'] == region].sort_values('year')
            color = cmap(idx % 10)
            plt.plot(region_data['year'], region_data['prod'], 
                    'o-', linewidth=2, markersize=4, label=region, 
                    color=color, alpha=0.8)
        
        last_year = historical_df['year'].max()
        plt.axvline(x=last_year, color='gray', linestyle='--', 
                   linewidth=1.5, alpha=0.5, label='Forecast Start')
    
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {filename_csv}")
    print(f"Saved: {filename_png}")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """
    Main execution pipeline:
    1. Load historical data
    2. For each region/country:
       a. Predict future climate trends
       b. Train model on historical climate-production relationships
       c. Use predicted climate to forecast production
    """
    
    # Configuration
    future_years = list(range(2023, 2043))
    
    # Load data
    regional_df, aggregated_df = load_data()
    
    # National forecasting
    national_predictions = forecast_national(aggregated_df, future_years)
    save_results(aggregated_df, national_predictions, level='national')
    
    # Regional forecasting
    regional_predictions = forecast_regional(regional_df, future_years)
    save_results(regional_df, regional_predictions, level='regional')
    
    print("\n" + "=" * 70)
    print("FORECASTING COMPLETE")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  - brazil_future_predictions.csv")
    print("  - brazil_forecast.png")
    print("  - regional_future_predictions.csv")
    print("  - regional_forecast.png")
    print("\nMethodology Summary:")
    print("  1. Climate variables predicted using linear regression")
    print("  2. Neural network trained on historical climate-production data")
    print("  3. Production forecasted based on predicted climate conditions")

if __name__ == "__main__":
    main()