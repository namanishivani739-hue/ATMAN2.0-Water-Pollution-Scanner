# ============================================================
#  WATER POLLUTION CLASSIFIER
#  Project: Scanning & Collection of Polluted Items in Water Bodies
#  Model: Random Forest Classifier
#  Goal: Predict pollutant type from sensor readings
# ============================================================

# ── STEP 1: Import libraries ─────────────────────────────────
# These are tools we borrow — no need to build from scratch
import pandas as pd                          # for tables (DataFrames)
import numpy as np                           # for numbers and math
import matplotlib.pyplot as plt              # for plotting charts
import seaborn as sns                        # for beautiful charts
from sklearn.ensemble import RandomForestClassifier   # our ML model
from sklearn.model_selection import train_test_split  # to split data
from sklearn.metrics import classification_report, confusion_matrix  # to evaluate
from sklearn.preprocessing import LabelEncoder        # to convert text to numbers
import joblib                                # to save the trained model
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  WATER POLLUTION CLASSIFIER — Starting...")
print("=" * 55)


# ── STEP 2: Create sample data ───────────────────────────────
# Since we don't have real sensor data yet, we simulate it.
# Each row = one water sample collected by your sensor/drone.
# Columns = what your sensors measure.
# When you collect real data, replace this block with:
#   df = pd.read_csv("your_real_data.csv")

np.random.seed(42)   # so results are same every time you run
n = 300              # 300 water samples

# Each pollutant type has different sensor "signatures"
def make_samples(pollutant, n_samples, turbidity_range, ph_range, temp_range, chemical_range, oxygen_range):
    return pd.DataFrame({
        'turbidity':       np.random.uniform(*turbidity_range, n_samples),   # how cloudy water is (NTU)
        'pH':              np.random.uniform(*ph_range, n_samples),           # acidity (0-14)
        'temperature':     np.random.uniform(*temp_range, n_samples),         # water temp (°C)
        'chemical_level':  np.random.uniform(*chemical_range, n_samples),     # chemical concentration (mg/L)
        'dissolved_oxygen':np.random.uniform(*oxygen_range, n_samples),       # oxygen in water (mg/L)
        'pollutant_type':  pollutant
    })

plastic_waste       = make_samples('Plastic Waste',        75, (20,80),  (6.5,7.5), (20,30), (1,10),   (6,9))
industrial_chemicals= make_samples('Industrial Chemicals', 75, (60,150), (3.0,6.0), (25,40), (50,200), (1,4))
algal_bloom         = make_samples('Algal Bloom',          75, (30,100), (8.0,10.0),(25,35), (5,30),   (3,7))
foam_detergent      = make_samples('Foam/Detergent',       75, (80,180), (7.5,9.5), (20,30), (20,80),  (2,5))

# Combine all samples into one table
df = pd.concat([plastic_waste, industrial_chemicals, algal_bloom, foam_detergent], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle rows

print(f"\n[STEP 2] Sample data created: {df.shape[0]} rows x {df.shape[1]} columns")
print(df.head(8).to_string(index=False))


# ── STEP 3: Explore the data (EDA) ──────────────────────────
print("\n[STEP 3] Data overview:")
print(df.describe().round(2).to_string())

print("\n  Pollutant counts:")
print(df['pollutant_type'].value_counts().to_string())

# Check for missing values
print(f"\n  Missing values: {df.isnull().sum().sum()} (none — good!)")


# ── STEP 4: Prepare data for the model ───────────────────────
# ML models only understand numbers, not text like "Plastic Waste"
# So we convert the labels to numbers: 0, 1, 2, 3

le = LabelEncoder()
df['label'] = le.fit_transform(df['pollutant_type'])

# Show what each number means
print("\n[STEP 4] Label encoding:")
for num, name in enumerate(le.classes_):
    print(f"  {num} → {name}")

# Separate features (X) from target (y)
# X = what we give the model (sensor readings)
# y = what we want to predict (pollutant type)
feature_cols = ['turbidity', 'pH', 'temperature', 'chemical_level', 'dissolved_oxygen']
X = df[feature_cols]   # input features
y = df['label']        # output labels

print(f"\n  Features (X) shape: {X.shape}")
print(f"  Labels   (y) shape: {y.shape}")


# ── STEP 5: Split data into Train and Test ────────────────────
# We never test on data the model has already seen — that's cheating!
# 80% for training, 20% for testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% goes to test
    random_state=42,     # so split is same every run
    stratify=y           # keeps class balance in both splits
)

print(f"\n[STEP 5] Train/Test split:")
print(f"  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")


# ── STEP 6: Train the model ───────────────────────────────────
# Random Forest = builds many decision trees and votes on the answer
# n_estimators = how many trees (more = more accurate but slower)

model = RandomForestClassifier(
    n_estimators=100,    # 100 decision trees
    max_depth=10,        # how deep each tree can grow
    random_state=42
)

print("\n[STEP 6] Training the model...")
model.fit(X_train, y_train)   # THIS is where learning happens
print("  Training complete!")


# ── STEP 7: Evaluate the model ────────────────────────────────
y_pred = model.predict(X_test)

accuracy = (y_pred == y_test).mean() * 100
print(f"\n[STEP 7] Model Accuracy: {accuracy:.1f}%")

print("\n  Detailed report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# ── STEP 8: Visualise results ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Water Pollution Classifier — Results", fontsize=14, fontweight='bold')

# Chart 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
axes[0].set_title("Confusion Matrix\n(diagonal = correct predictions)")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")
axes[0].tick_params(axis='x', rotation=30)

# Chart 2: Feature Importance (which sensor matters most?)
importances = model.feature_importances_
feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=True)
axes[1].barh(feat_df['Feature'], feat_df['Importance'], color='steelblue')
axes[1].set_title("Feature Importance\n(which sensor matters most?)")
axes[1].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/pollution_results.png', dpi=150, bbox_inches='tight')
print("\n[STEP 8] Charts saved!")


# ── STEP 9: Make a prediction on new data ────────────────────
# This is how you'd use it in your real project:
# Give it sensor readings → it tells you what type of pollution

print("\n[STEP 9] Predicting on new sensor readings...")
new_sample = pd.DataFrame({
    'turbidity':        [95.0],   # high turbidity
    'pH':               [4.5],    # very acidic
    'temperature':      [32.0],
    'chemical_level':   [120.0],  # high chemicals
    'dissolved_oxygen': [2.1]     # very low oxygen
})

prediction = model.predict(new_sample)
probabilities = model.predict_proba(new_sample)[0]

predicted_name = le.inverse_transform(prediction)[0]
print(f"\n  Sensor input  : turbidity=95, pH=4.5, temp=32, chemical=120, O2=2.1")
print(f"  Prediction    : {predicted_name}")
print(f"\n  Confidence for each type:")
for name, prob in zip(le.classes_, probabilities):
    bar = "█" * int(prob * 30)
    print(f"    {name:<25} {prob*100:5.1f}%  {bar}")


# ── STEP 10: Save the model ───────────────────────────────────
# Save so you can load it later without retraining
joblib.dump(model, '/mnt/user-data/outputs/pollution_classifier.pkl')
joblib.dump(le,    '/mnt/user-data/outputs/label_encoder.pkl')
print("\n[STEP 10] Model saved as 'pollution_classifier.pkl'")
print("          To load later: model = joblib.load('pollution_classifier.pkl')")

print("\n" + "=" * 55)
print("  DONE! Your ML model is trained and saved.")
print("  Next: collect real sensor data and replace Step 2.")
print("=" * 55)
