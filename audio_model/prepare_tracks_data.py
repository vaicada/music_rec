import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
import gc

print("Starting data preprocessing for Audio Autoencoder (Model 2)...")

# Create directories if they don't exist
os.makedirs('data/processed/tracks', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. Load data
csv_path = 'dataset/tracks_features.csv'
print(f"Loading {csv_path} (This might take a minute for 1.2M rows...)")
df = pd.read_csv(csv_path)

# 2. Rename columns to match expected schema if necessary
# In tracks_features.csv, columns are normally already named 'energy', 'danceability', etc.
rename_map = {
    'name': 'song_name',
    'artists': 'artist'
}
df = df.rename(columns=rename_map)

# Target audio features for the Autoencoder
AUDIO_COLS_RAW = [
    'energy', 'danceability', 'valence', 'tempo',
    'acousticness', 'instrumentalness', 'speechiness', 'liveness', 'key'
]

# Ensure all required columns exist
missing_cols = [c for c in AUDIO_COLS_RAW if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

# 3. Handle Nulls
print(f"Original size: {len(df)}")
df = df.dropna(subset=AUDIO_COLS_RAW)
print(f"Size after dropping audio nulls: {len(df)}")

# 4. Normalize Audio Features
# Most features are 0-1, but tempo is BPM and key is 0-11
SCALE_100 = ['energy', 'danceability', 'valence', 
             'acousticness', 'instrumentalness', 'speechiness', 'liveness']

for col in SCALE_100:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # If using 0-100 scale, convert to 0-1
    if df[col].max() > 1.5:  # giving some buffer over 1.0 just in case
        print(f"Scaling column {col} by /100")
        df[col] = df[col] / 100.0

df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')

# Drop any newly introduced nulls from numeric conversion
df = df.dropna(subset=SCALE_100 + ['tempo', 'key'])
print(f"Size after numeric conversion nulls: {len(df)}")

# 5. Encode Key (0-11 to 0-1)
# Typically keys are integers 0-11. 
df['key'] = pd.to_numeric(df['key'], errors='coerce')
df['key'] = df['key'] / 11.0 # Scale to 0-1

# Verify final size
df = df.dropna(subset=AUDIO_COLS_RAW)
print(f"Final valid dataset size: {len(df)}")

# 6. Split data (90/5/5 for a huge dataset like this)
print("Splitting data into train/val/test...")
# We do not stratify because we don't have emotion labels anymore!
train, temp = train_test_split(df, test_size=0.10, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

def save_split(split_df, split_name):
    # Extract only the 9 features as float32 numpy arrays to save space
    features = split_df[AUDIO_COLS_RAW].values.astype(np.float32)
    
    # Save metadata separately (song_id, name, artist)
    # tracks_features usually has 'id'
    meta_cols = ['id', 'song_name', 'artist', 'year', 'album']
    available_meta = [c for c in meta_cols if c in split_df.columns]
    
    meta_df = split_df[available_meta]
    
    np.save(f'data/processed/tracks/features_{split_name}.npy', features)
    meta_df.to_parquet(f'data/processed/tracks/meta_{split_name}.parquet', index=False)
    print(f"Saved {split_name} - {len(split_df)} records")

save_split(train, 'train')
save_split(val, 'val')
save_split(test, 'test')

# 7. Compute and save stats for Z-score normalization (Tempo mainly needs this)
print("Computing normalization stats from train set...")
# We use only train to compute mean/std to avoid data leakage
train_features = train[AUDIO_COLS_RAW]
stats = {
    'mean': train_features.mean().tolist(),
    'std': train_features.std().tolist(),
    'features': AUDIO_COLS_RAW,
    'note': 'Used for Autoencoder Inference Z-score normalization'
}

with open('models/tracks_stats.json', 'w') as f:
    json.dump(stats, f, indent=4)

print("Saved models/tracks_stats.json")

# Clean up memory
del df, train, val, test, temp
gc.collect()

print("Preprocessing completed successfully! Data ready for Autoencoder training.")
