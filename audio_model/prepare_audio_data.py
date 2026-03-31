import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split

print("Starting data preprocessing for Model 2...")

# Create directories if they don't exist
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. Load data
print("Loading spotify_dataset.csv...")
df = pd.read_csv('dataset/spotify_dataset.csv')

# 2. Rename columns
rename_map = {
    'Artist(s)': 'artist',
    'song': 'song_name',
    'Energy': 'energy',
    'Danceability': 'danceability',
    'Positiveness': 'valence',
    'Tempo': 'tempo',
    'Acousticness': 'acousticness',
    'Instrumentalness': 'instrumentalness',
    'Speechiness': 'speechiness',
    'Liveness': 'liveness',
    'Key': 'key'
}
df = df.rename(columns=rename_map)

AUDIO_COLS_RAW = ['energy', 'danceability', 'valence', 'tempo',
                  'acousticness', 'instrumentalness', 'speechiness', 'liveness', 'key']

# 3. Drop nulls in audio columns
print(f"Original size: {len(df)}")
df = df.dropna(subset=AUDIO_COLS_RAW)
df = df.dropna(subset=['emotion'])
print(f"Size after dropping audio/emotion nulls: {len(df)}")

# 4. Normalize 0-100 to 0-1
SCALE_100 = ['energy', 'danceability', 'valence',
             'acousticness', 'instrumentalness', 'speechiness', 'liveness']

for col in SCALE_100:
    # Convert to numeric, handle errors
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # Ifmax > 1, assume 0-100 scale and divide by 100
    if df[col].max() > 1:
        df[col] = df[col] / 100.0

df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')

# Drop nulls that might have been introduced by numeric conversion
df = df.dropna(subset=SCALE_100 + ['tempo'])
print(f"Size after numeric conversion nulls: {len(df)}")


# 5. Encode Key
KEY_MAP = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7,
    'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

def parse_key(key_str):
    if pd.isna(key_str):
        return 5 / 11.0
    root = str(key_str).split()[0]
    return KEY_MAP.get(root, 5) / 11.0

df['key'] = df['key'].apply(parse_key)

# 6. Encode Emotion
EMOTION_MAP = {
    'joy': 0, 'sadness': 1, 'anger': 2,
    'fear': 3, 'love': 4, 'surprise': 5
}
df['emotion_label'] = df['emotion'].str.lower().map(EMOTION_MAP)
df = df.dropna(subset=['emotion_label'])
df['emotion_label'] = df['emotion_label'].astype(int)
print(f"Final valid dataset size: {len(df)}")

# 7. Split data (80/10/10) stratified
print("Splitting data into train/val/test...")
train, temp = train_test_split(df, test_size=0.2, stratify=df['emotion_label'], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp['emotion_label'], random_state=42)

# Save to CSV
cols_to_save = ['artist', 'song_name', 'emotion', 'emotion_label'] + AUDIO_COLS_RAW
train[cols_to_save].to_csv('data/processed/audio_train.csv', index=False)
val[cols_to_save].to_csv('data/processed/audio_val.csv', index=False)
test[cols_to_save].to_csv('data/processed/audio_test.csv', index=False)

print(f"Train size: {len(train)}")
print(f"Val size: {len(val)}")
print(f"Test size: {len(test)}")

# 8. Compute and save stats for normalization
# Calculate mean and std from TRAIN set only
stats_features = ['energy', 'danceability', 'valence', 'tempo',
                  'acousticness', 'instrumentalness', 'speechiness', 'liveness', 'key']

stats = {
    'mean': train[stats_features].mean().tolist(),
    'std': train[stats_features].std().tolist(),
    'features': stats_features
}

with open('models/audio2_stats.json', 'w') as f:
    json.dump(stats, f, indent=4)

print("Saved audio2_stats.json")
print("Preprocessing completed successfully!")
