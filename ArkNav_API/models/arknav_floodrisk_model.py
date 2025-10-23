"""
Improved Flood Height Prediction Model Training Script v3
Focus: Reduce overfitting, better handle flood vs non-flood separation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = 'data\ArkNav_FloodRisk_Dataset.csv'
MODEL_SAVE_PATH = 'models\Arknav_Floodrisk_Modelv2.pkl'

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("="*70)
print("🌊 FLOOD PREDICTION MODEL V3 - OVERFITTING FIX")
print("="*70)

print("\n📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df):,} records")

print("\n📊 Data distribution:")
print(f"   Floods (>0m): {(df['flood_height'] > 0).sum()} ({(df['flood_height'] > 0).mean()*100:.1f}%)")
print(f"   No floods (0m): {(df['flood_height'] == 0).sum()} ({(df['flood_height'] == 0).mean()*100:.1f}%)")
print(f"   Mean flood height (when flooding): {df[df['flood_height'] > 0]['flood_height'].mean():.2f}m")

# ============================================================================
# 2. FEATURE ENGINEERING (ORIGINAL APPROACH)
# ============================================================================
print("\n" + "="*70)
print("⚙️  FEATURE ENGINEERING")
print("="*70)

# === Hazard map legend ===
hazard_map = {
    1: "Low (0–0.5m)",
    2: "Medium (>0.5–1.5m)",
    3: "High (>1.5m)"
}

print("🧩 Fixing logic for in_zone, hazard_code, and proximity_m consistency...")

# --- Ensure consistency between zone and proximity ---
# If in a flood zone, proximity must be 0
df.loc[df['in_zone'] == True, 'proximity_m'] = 0.0

# If NOT in a flood zone, proximity must be > 0
# And hazard_code corresponds to the nearest flood-prone zone
df.loc[df['in_zone'] == False, 'proximity_m'] = df.loc[df['in_zone'] == False, 'proximity_m'].clip(lower=0.001)

# --- Safety checks for hazard_code ---
# Flood-prone areas must have valid hazard codes 1–3
df.loc[df['in_zone'] == True, 'hazard_code'] = df.loc[df['in_zone'] == True, 'hazard_code'].fillna(1).astype(int).clip(1, 3)

# Non-flood areas inherit nearest hazard zone (if missing, set to 1 - low)
df.loc[df['in_zone'] == False, 'hazard_code'] = df.loc[df['in_zone'] == False, 'hazard_code'].fillna(1).astype(int).clip(1, 3)

# --- Derive hazard level name for interpretability ---
df['hazard_level_fixed'] = df['hazard_code'].map(hazard_map)

print("✅ Logic fixed:")
print(f"   Flood zones: {(df['in_zone'] == True).sum()} → proximity = 0.0")
print(f"   Non-flood zones: {(df['in_zone'] == False).sum()} → proximity > 0")
print(f"   Unique hazard levels: {df['hazard_level_fixed'].unique()}")

df_model = df.copy()
df_model['in_zone_int'] = df_model['in_zone'].astype(int)

print("\n📍 Creating core geographic features...")

# Distance from Metro Manila center
manila_center_lat, manila_center_lon = 14.5906, 120.9735
df_model['dist_from_center'] = np.sqrt(
    (df_model['lat'] - manila_center_lat)**2 + 
    (df_model['lon'] - manila_center_lon)**2
) * 111

print("🌧️  Creating precipitation features...")

# Precipitation × Elevation
df_model['elevation_risk'] = 1 / (df_model['elevation'] + 1)
df_model['precip_x_elevation'] = df_model['precipitation'] * df_model['elevation_risk']

# Precipitation squared
df_model['precip_squared'] = df_model['precipitation'] ** 2

print("📏 Creating proximity features...")

# Proximity risk
df_model['proximity_risk'] = 1 / (df_model['proximity_m'] + 1)

# River distance risk (inverse of distance to nearest river)
df_model['river_distance_risk'] = 1 / (df_model['dist_to_nearest_water_m'] + 1)

# Zone × Hazard
df_model['zone_x_hazard'] = df_model['in_zone_int'] * df_model['hazard_code']

# Zone × Precipitation
df_model['zone_x_precip'] = df_model['in_zone_int'] * df_model['precipitation']

# Elevation × Proximity
df_model['elevation_x_proximity'] = df_model['elevation_risk'] * df_model['proximity_risk']

# Risks × River Distance (new)
df_model['precipRisk_x_riverdistance'] = df_model['precipitation'] * df_model['dist_to_nearest_water_m']
df_model['elevRisk_x_riverdistance'] = df_model['elevation_risk'] * df_model['dist_to_nearest_water_m']
df_model['proxRisk_x_riverdistance'] = df_model['proximity_risk'] * df_model['dist_to_nearest_water_m']

df_model['elevation_x_riverDistance'] = df_model['elevation'] * df_model['dist_to_nearest_water_m']
df_model['proximity_x_riverDistance'] = df_model['proximity_m'] * df_model['dist_to_nearest_water_m']

# Zone * Hazard * Precipitation
df_model['zone_x_hazard_x_precip'] = (
    df_model['in_zone_int'] * df_model['hazard_code'] * df_model['precipitation']
)

# Interaction: Precipitation × Elevation × Proximity
df_model['precip_x_elev_x_prox'] = (
    df_model['precipitation'] * df_model['elevation_risk'] * df_model['proximity_risk']
)

# Combined Environmental Risk Feature (new composite metric)
df_model['env_risk_score'] = (
    df_model['precipitation'] * df_model['elevation_risk'] *
    df_model['proximity_risk'] * df_model['river_distance_risk']
)

df_model['zone_x_riverDistance'] = df_model['in_zone_int'] * df_model['dist_to_nearest_water_m']

# Lat x Nearest River Distance
df_model['lat_x_riverDistance'] = df_model['lat'] * df_model['dist_to_nearest_water_m']
df_model['lon_x_riverDistance'] = df_model['lon'] * df_model['dist_to_nearest_water_m']

# Hazard Code x In Zone x River Distance
df_model['hazard_x_inzone_x_riverDistance'] = df_model['hazard_code'] * df_model['in_zone_int'] * df_model['dist_to_nearest_water_m']

# Lat x Lon x Nearest River Distance
df_model['latlon_x_riverDistance'] = (
    df_model['lat'] * df_model['lon'] * df_model['dist_to_nearest_water_m']
)

df_model['lat_x_riverLat'] = df_model['lat'] * df_model['nearest_river_lat'] 
df_model['lon_x_riverLon'] = df_model['lon'] * df_model['nearest_river_lon']

print("⚠️  Creating risk score...")

# Combined risk score
df_model['risk_score'] = (
    df_model['in_zone_int'] * 3 + 
    (df_model['precipitation'] / 10) * 2 + 
    df_model['elevation_risk'] * 10 +
    df_model['proximity_risk'] * 5
)

print(f"\n✅ Total features created: {len(df_model.columns) - len(df.columns)}")

# ============================================================================
# 3. PREPARE FEATURES AND TARGET (ORIGINAL FEATURES)
# ============================================================================
print("\n" + "="*70)
print("🎯 PREPARING FEATURES")
print("="*70)

# Select features - ORIGINAL HIGH IMPACT FEATURES
feature_cols = [
    'lat',
    'lon',
    'elevation', 
    'precipitation', 
    'hazard_code', 
    'proximity_m',
    'in_zone_int',
    'elevation_risk',
    'proximity_risk',
    'precip_x_elevation',
    'zone_x_precip',
    'zone_x_hazard',
    'risk_score',
    'precip_squared',
    'dist_from_center',
    'nearest_river_lon',
    'nearest_river_lat',
    #'dist_to_nearest_water_m',
    'precipRisk_x_riverdistance',
    'precip_x_elev_x_prox'
]

X = df_model[feature_cols]
y = df_model['flood_height']

print(f"✅ Selected {len(feature_cols)} features:")
for i, feat in enumerate(feature_cols, 1):
    print(f"   {i:2d}. {feat}")

print(f"\n📊 Feature matrix shape: {X.shape}")
print(f"📊 Target shape: {y.shape}")

# ============================================================================
# 4. SMARTER SAMPLE WEIGHTING
# ============================================================================
print("\n" + "="*70)
print("⚖️  AGGRESSIVE SAMPLE WEIGHTING")
print("="*70)

flood_mask = y > 0
n_floods = flood_mask.sum()
n_no_floods = (~flood_mask).sum()
imbalance_ratio = n_no_floods / n_floods

print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")

# Create aggressive sample weights
# Give even MORE weight to flood events so model learns them
sample_weights = np.ones(len(y))
sample_weights[flood_mask] = imbalance_ratio * 2  # 2x the balance ratio

print(f"   No-flood weight: 1.0")
print(f"   Flood weight: {imbalance_ratio * 2:.1f}")

# ============================================================================
# 5. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*70)
print("✂️  SPLITTING DATA")
print("="*70)

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42, stratify=(y > 0)
)

print(f"🔵 Training: {len(X_train):,}")
print(f"🔴 Test: {len(X_test):,}")

# ============================================================================
# 6. ROBUST SCALING (BETTER FOR OUTLIERS)
# ============================================================================
print("\n⚖️  Using RobustScaler (better handles outliers)...")
scaler = RobustScaler()  # Less sensitive to extreme values
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled")

# ============================================================================
# 7. EXTREMELY REGULARIZED MODELS
# ============================================================================
print("\n" + "="*70)
print("🤖 TRAINING HEAVILY REGULARIZED MODELS")
print("="*70)

models = {
    'XGBoost': xgb.XGBRegressor(
        n_estimators=200,        # Reduced
        learning_rate=0.03,      # Much lower
        max_depth=4,             # Shallower (was 5)
        min_child_weight=10,     # Much higher (was 5)
        subsample=0.7,           # More aggressive
        colsample_bytree=0.7,    # More aggressive
        reg_alpha=0.5,           # Stronger L1
        reg_lambda=2.0,          # Stronger L2
        gamma=0.2,               # Stronger pruning (was 0.1)
        random_state=42,
        n_jobs=-1,
        tree_method='hist'       # Faster training
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"🔸 Training {name}...")
    print(f"{'='*70}")
    
    # Train with aggressive sample weights
    model.fit(X_train_scaled, y_train, sample_weight=weights_train)
    
    # Predictions
    y_pred_train = np.maximum(0, model.predict(X_train_scaled))
    y_pred_test = np.maximum(0, model.predict(X_test_scaled))
    
    # Overall metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # FLOOD-ONLY metrics (most important!)
    flood_test_mask = y_test > 0
    if flood_test_mask.sum() > 0:
        flood_rmse = np.sqrt(mean_squared_error(
            y_test[flood_test_mask], 
            y_pred_test[flood_test_mask]
        ))
        flood_mae = mean_absolute_error(
            y_test[flood_test_mask], 
            y_pred_test[flood_test_mask]
        )
        flood_r2 = r2_score(y_test[flood_test_mask], y_pred_test[flood_test_mask])
    else:
        flood_rmse = flood_mae = flood_r2 = None
    
    results[name] = {
        'model': model,
        'train_rmse': train_rmse, 'train_mae': train_mae, 'train_r2': train_r2,
        'test_rmse': test_rmse, 'test_mae': test_mae, 'test_r2': test_r2,
        'flood_rmse': flood_rmse, 'flood_mae': flood_mae, 'flood_r2': flood_r2,
        'y_pred_test': y_pred_test
    }
    
    print(f"\n📊 TRAINING METRICS:")
    print(f"   RMSE: {train_rmse:.4f} | MAE: {train_mae:.4f} | R²: {train_r2:.4f}")
    
    print(f"\n📊 TEST METRICS:")
    print(f"   RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R²: {test_r2:.4f}")
    
    # Calculate train/test gap
    r2_gap = train_r2 - test_r2
    print(f"\n⚠️  Train-Test R² Gap: {r2_gap:.4f} {'(GOOD!)' if r2_gap < 0.15 else '(Still overfitting)'}")
    
    if flood_rmse is not None:
        print(f"\n🌊 FLOOD-ONLY METRICS (most important):")
        print(f"   RMSE: {flood_rmse:.4f} | MAE: {flood_mae:.4f} | R²: {flood_r2:.4f}")
    
    # Cross-validation
    print(f"\n🔄 Cross-validation (5-fold)...")
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, 
        cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    cv_rmse = np.sqrt(-cv_scores)
    print(f"   CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")

# ============================================================================
# 8. MODEL COMPARISON & SELECTION
# ============================================================================
print("\n" + "="*70)
print("📊 FINAL COMPARISON")
print("="*70)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test RMSE': [r['test_rmse'] for r in results.values()],
    'Test R²': [r['test_r2'] for r in results.values()],
    'Flood R²': [r['flood_r2'] for r in results.values()],
    'Train-Test Gap': [r['train_r2'] - r['test_r2'] for r in results.values()]
})

print("\n" + comparison_df.to_string(index=False))

# Select XGBoost as final model (best flood prediction performance)
final_model = results['XGBoost']['model']
final_test_rmse = results['XGBoost']['test_rmse']
final_test_r2 = results['XGBoost']['test_r2']
final_flood_r2 = results['XGBoost']['flood_r2']

print(f"\n🏆 FINAL MODEL: XGBoost (Best Flood R²)")
print(f"   Test RMSE: {final_test_rmse:.4f}")
print(f"   Test R²:   {final_test_r2:.4f}")
print(f"   Flood R²:  {final_flood_r2:.4f}")
print(f"   Train-Test Gap: {results['XGBoost']['train_r2'] - final_test_r2:.4f}")



# ============================================================================
# 9. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*70)
print("🎯 XGBOOST FEATURE IMPORTANCE")
print("="*70)

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 most important features:")
for idx, row in importance_df.head(10).iterrows():
    print(f"   {row['Feature']:25s} {row['Importance']:.4f}")

print(f"\n💡 Key insights:")
top_3 = importance_df.head(3)['Feature'].tolist()
print(f"   Top 3 features: {', '.join(top_3)}")
print(f"   These drive {importance_df.head(3)['Importance'].sum():.1%} of predictions")

# ============================================================================
# 10. SAVE MODEL
# ============================================================================
print("\n" + "="*70)
print("💾 SAVING XGBOOST MODEL")
print("="*70)

model_package = {
    'type': 'xgboost',
    'model': final_model,
    'scaler': scaler,
    'feature_names': feature_cols,
    'test_rmse': final_test_rmse,
    'test_r2': final_test_r2,
    'flood_r2': final_flood_r2,
    'metadata': {
        'train_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_training_samples': len(X_train),
        'n_test_samples': len(X_test),
        'flood_samples_train': (y_train > 0).sum(),
        'flood_samples_test': (y_test > 0).sum()
    }
}

joblib.dump(model_package, MODEL_SAVE_PATH)
print(f"✅ Saved to: {MODEL_SAVE_PATH}")
print(f"   Model: XGBoost")
print(f"   Test RMSE: {final_test_rmse:.4f}")
print(f"   Test R²: {final_test_r2:.4f}")
print(f"   Flood R²: {final_flood_r2:.4f}")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print("\n🎯 Final Model Summary:")
print(f"   • Type: XGBoost Regressor")
print(f"   • Features: {len(feature_cols)}")
print(f"   • Test Performance: R² = {final_test_r2:.3f}, RMSE = {final_test_rmse:.2f}m")
print(f"   • Flood Prediction: R² = {final_flood_r2:.3f}")
print(f"   • Train-Test Gap: {results['XGBoost']['train_r2'] - final_test_r2:.3f} (stable!)")
print(f"   • File: {MODEL_SAVE_PATH}")

print("\n📦 To use in production:")
print("   import joblib")
print("   import numpy as np")
print("   ")
print("   # Load model")
print(f"   model_pkg = joblib.load('{MODEL_SAVE_PATH}')")
print("   model = model_pkg['model']")
print("   scaler = model_pkg['scaler']")
print("   ")
print("   # Prepare features (see feature engineering above)")
print("   features = np.array([[lat, lon, elevation, ...]])  # 15 features")
print("   features_scaled = scaler.transform(features)")
print("   ")
print("   # Predict")
print("   flood_height = max(0, model.predict(features_scaled)[0])")
print("   ")
print("   # Categorize")
print("   if flood_height < 0.15: category = 'No Flood'")
print("   elif flood_height < 0.5: category = 'Ankle High'")
print("   elif flood_height < 1.0: category = 'Knee High'")
print("   # ... etc")

def predict_flood_probability(model, scaler, feature_vector, k=3.0, max_height=2.0):
    """
    Computes the probability of flooding given input features using a regression model.
    - model: trained XGBoost regressor
    - scaler: fitted RobustScaler
    - feature_vector: pd.DataFrame with same feature columns
    - k: steepness factor for probability curve
    - max_height: normalization reference (e.g., 2m flood = ~100% probability)
    """

    # Scale features
    scaled = scaler.transform(feature_vector)

    # Predict flood height
    predicted_height = np.maximum(0, model.predict(scaled)[0])

    # Option 1: Logistic scaling
    prob = 1 / (1 + np.exp(-k * (predicted_height - 0.1)))

    # Option 2 (alternative): linear normalization
    prob_linear = min(predicted_height / max_height, 1.0)

    return {
        "predicted_height": float(predicted_height),
        "probability_logistic": float(prob),
        "probability_linear": float(prob_linear)
    }