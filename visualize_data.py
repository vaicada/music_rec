import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Cấu hình
DATA_PATH = r"data/processed/train.csv"
OUTPUT_DIR = "visualizations"
AUDIO_FEATURES = [
    'energy', 'danceability', 'valence', 
    'acousticness', 'instrumentalness', 
    'speechiness', 'liveness', 'tempo'
]
CONTEXT_COLS = [
    'Good for Driving', 'Good for Yoga/Stretching', 
    'Good for Morning Routine', 'Good for Social Gatherings',
    'Good for Work/Study', 'Good for Exercise',
    'Good for Relaxation/Meditation', 'Good for Party', 
    'Good for Running'
]
NUMERIC_FEATURES = ['popularity', 'Length']
CATEGORICAL_FEATURES = ['key', 'Time signature', 'Explicit']

def setup_style():
    """Thiết lập style cho biểu đồ đẹp hơn"""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 12

def load_data():
    """Load dữ liệu train"""
    print(f"Loading data from {DATA_PATH}...")
    # Chỉ load các cột cần thiết để tiết kiệm RAM
    cols = ['emotion', 'genre', 'Release Date'] + AUDIO_FEATURES + CONTEXT_COLS + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    
    # Do file lớn, ta chỉ đọc 1 phần mẫu nếu cần, hoặc đọc hết
    # Ở đây đọc hết vì server có 16GB RAM khá thoải mái
    df = pd.read_csv(DATA_PATH, usecols=cols)
    print(f"Loaded {len(df)} rows.")
    return df

def plot_emotion_distribution(df):
    """Vẽ phân bố cảm xúc"""
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='emotion', order=df['emotion'].value_counts().index, palette='viridis')
    plt.title('Phân Bố Số Lượng Bài Hát Theo Cảm Xúc (Emotion)')
    plt.xlabel('Cảm Xúc')
    plt.ylabel('Số Lượng')
    plt.bar_label(ax.containers[0])
    plt.savefig(f"{OUTPUT_DIR}/emotion_distribution.png")
    plt.close()
    print("Saved emotion_distribution.png")

def plot_audio_features_boxplot(df):
    """Vẽ boxplot cho các đặc trưng âm thanh"""
    plt.figure(figsize=(14, 8))
    
    # Melt dataframe để vẽ boxplot gộp
    df_melted = df[AUDIO_FEATURES].melt(var_name='Feature', value_name='Value')
    
    sns.boxplot(data=df_melted, x='Feature', y='Value', palette='Set3')
    plt.title('Phân Bố Giá Trị Các Đặc Trưng Âm Thanh (Audio Features)')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.1)  # Các feature đa số scale 0-1
    plt.savefig(f"{OUTPUT_DIR}/audio_features_boxplot.png")
    plt.close()
    print("Saved audio_features_boxplot.png")

def plot_audio_features_histogram(df):
    """Vẽ histogram cho từng feature"""
    plt.figure(figsize=(20, 10))
    for i, feature in enumerate(AUDIO_FEATURES):
        plt.subplot(2, 4, i+1)
        sns.histplot(df[feature], bins=30, kde=True, color='skyblue')
        plt.title(f'Phân Bố {feature.capitalize()}')
        plt.xlabel(feature)
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/audio_features_histogram.png")
    plt.close()
    print("Saved audio_features_histogram.png")

def plot_correlation_matrix(df):
    """Vẽ ma trận tương quan giữa các features"""
    plt.figure(figsize=(10, 8))
    corr = df[AUDIO_FEATURES].corr()
    
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title('Ma Trận Tương Quan (Correlation Matrix)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_matrix.png")
    plt.close()
    print("Saved correlation_matrix.png")

def plot_context_tags(df):
    """Vẽ thống kê các thẻ ngữ cảnh"""
    # Tính tổng số lượng True (1) cho mỗi cột context
    # Lưu ý: Cần convert sang numeric nếu chưa phải
    context_counts = df[CONTEXT_COLS].apply(pd.to_numeric, errors='coerce').sum().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=context_counts.values, y=context_counts.index, palette='magma')
    plt.title('Số Lượng Bài Hát Cho Từng Ngữ Cảnh (Context)')
    plt.xlabel('Số Lượng Bài Hát')
    plt.savefig(f"{OUTPUT_DIR}/context_distribution.png")
    plt.close()
    print("Saved context_distribution.png")

def plot_genre_distribution(df):
    """Vẽ top 20 genres phổ biến nhất"""
    plt.figure(figsize=(12, 10))
    top_genres = df['genre'].value_counts().head(20)
    
    sns.barplot(x=top_genres.values, y=top_genres.index, palette='rocket')
    plt.title('Top 20 Thể Loại Nhạc Phổ Biến Nhất')
    plt.xlabel('Số Lượng')
    plt.savefig(f"{OUTPUT_DIR}/genre_distribution.png")
    plt.close()
    print("Saved genre_distribution.png")

def plot_numeric_features(df):
    """Vẽ histogram cho popularity và Length"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Popularity
    axes[0].hist(df['popularity'].dropna(), bins=50, color='teal', edgecolor='black', alpha=0.7)
    axes[0].set_title('Phân Bố Độ Phổ Biến (Popularity)', fontsize=14)
    axes[0].set_xlabel('Popularity')
    axes[0].set_ylabel('Count')
    axes[0].grid(alpha=0.3)
    
    # Length (convert to minutes)
    length_minutes = pd.to_numeric(df['Length'], errors='coerce') / 60000  # ms to minutes
    axes[1].hist(length_minutes.dropna(), bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_title('Phân Bố Độ Dài Bài Hát (Minutes)', fontsize=14)
    axes[1].set_xlabel('Length (minutes)')
    axes[1].set_ylabel('Count')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/numeric_features.png")
    plt.close()
    print("Saved numeric_features.png")

def plot_categorical_features(df):
    """Vẽ phân bố cho key, time signature, và explicit"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Key distribution
    key_counts = df['key'].value_counts().sort_index()
    axes[0].bar(key_counts.index.astype(str), key_counts.values, color='purple', alpha=0.7)
    axes[0].set_title('Phân Bố Key (Nốt Nhạc)', fontsize=14)
    axes[0].set_xlabel('Key')
    axes[0].set_ylabel('Count')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Time signature
    ts_counts = df['Time signature'].value_counts().sort_index()
    axes[1].bar(ts_counts.index.astype(str), ts_counts.values, color='orange', alpha=0.7)
    axes[1].set_title('Phân Bố Time Signature', fontsize=14)
    axes[1].set_xlabel('Time Signature')
    axes[1].set_ylabel('Count')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Explicit
    explicit_counts = df['Explicit'].value_counts()
    axes[2].pie(explicit_counts.values, labels=explicit_counts.index, autopct='%1.1f%%', 
                colors=['lightgreen', 'lightcoral'], startangle=90)
    axes[2].set_title('Tỷ Lệ Bài Hát Explicit', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/categorical_features.png")
    plt.close()
    print("Saved categorical_features.png")

def plot_release_date_trends(df):
    """Vẽ xu hướng phát hành theo năm"""
    plt.figure(figsize=(14, 6))
    
    # Extract year from Release Date
    df_copy = df.copy()
    df_copy['Release Date'] = pd.to_datetime(df_copy['Release Date'], errors='coerce')
    df_copy['Year'] = df_copy['Release Date'].dt.year
    
    # Count by year
    year_counts = df_copy['Year'].value_counts().sort_index()
    
    plt.plot(year_counts.index, year_counts.values, marker='o', linewidth=2, markersize=4, color='steelblue')
    plt.fill_between(year_counts.index, year_counts.values, alpha=0.3, color='steelblue')
    plt.title('Xu Hướng Phát Hành Bài Hát Theo Năm', fontsize=16)
    plt.xlabel('Năm')
    plt.ylabel('Số Lượng Bài Hát')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/release_date_trends.png")
    plt.close()
    print("Saved release_date_trends.png")

def plot_energy_valence_scatter(df):
    """Vẽ scatter plot Energy vs Valence, color by emotion"""
    plt.figure(figsize=(12, 8))
    
    # Sample data if too large
    sample_size = min(10000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    emotions = df_sample['emotion'].unique()
    colors = sns.color_palette('tab10', n_colors=len(emotions))
    
    for i, emotion in enumerate(emotions):
        subset = df_sample[df_sample['emotion'] == emotion]
        plt.scatter(subset['energy'], subset['valence'], 
                   label=emotion, alpha=0.5, s=20, c=[colors[i]])
    
    plt.title('Mối Quan Hệ Energy vs Valence (theo Emotion)', fontsize=16)
    plt.xlabel('Energy')
    plt.ylabel('Valence')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/energy_valence_scatter.png", dpi=150)
    plt.close()
    print("Saved energy_valence_scatter.png")

def plot_tempo_energy_scatter(df):
    """Vẽ scatter plot Tempo vs Energy"""
    plt.figure(figsize=(10, 7))
    
    # Sample data
    sample_size = min(10000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    plt.scatter(df_sample['tempo'], df_sample['energy'], 
               alpha=0.3, s=15, c=df_sample['danceability'], cmap='viridis')
    plt.colorbar(label='Danceability')
    plt.title('Mối Quan Hệ Tempo vs Energy (màu: Danceability)', fontsize=16)
    plt.xlabel('Tempo (BPM)')
    plt.ylabel('Energy')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tempo_energy_scatter.png", dpi=150)
    plt.close()
    print("Saved tempo_energy_scatter.png")

def plot_audio_by_emotion_violin(df):
    """Vẽ violin plot cho energy, valence, danceability theo emotion"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    features_to_plot = ['energy', 'valence', 'danceability']
    
    for i, feature in enumerate(features_to_plot):
        sns.violinplot(data=df, x='emotion', y=feature, ax=axes[i], palette='muted')
        axes[i].set_title(f'{feature.capitalize()} theo Emotion', fontsize=14)
        axes[i].set_xlabel('Emotion')
        axes[i].set_ylabel(feature.capitalize())
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/audio_by_emotion_violin.png", dpi=150)
    plt.close()
    print("Saved audio_by_emotion_violin.png")

def plot_popularity_by_genre(df):
    """Vẽ top 15 genre theo popularity trung bình"""
    plt.figure(figsize=(12, 8))
    
    # Calculate mean popularity by genre
    genre_pop = df.groupby('genre')['popularity'].mean().sort_values(ascending=False).head(15)
    
    sns.barplot(x=genre_pop.values, y=genre_pop.index, palette='crest')
    plt.title('Top 15 Thể Loại Có Popularity Trung Bình Cao Nhất', fontsize=16)
    plt.xlabel('Popularity Trung Bình')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/popularity_by_genre.png")
    plt.close()
    print("Saved popularity_by_genre.png")

def plot_tempo_distribution(df):
    """Vẽ phân bố tempo với KDE"""
    plt.figure(figsize=(12, 6))
    
    sns.histplot(df['tempo'].dropna(), bins=50, kde=True, color='darkblue', edgecolor='black')
    plt.title('Phân Bố Tempo (BPM)', fontsize=16)
    plt.xlabel('Tempo (BPM)')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tempo_distribution.png")
    plt.close()
    print("Saved tempo_distribution.png")

def print_dataset_summary(df):
    """In ra thống kê tổng quan về dataset"""
    print("\n" + "="*60)
    print("DATASET SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nTong so bai hat: {len(df):,}")
    print(f"So luong emotion duy nhat: {df['emotion'].nunique()}")
    print(f"So luong genre duy nhat: {df['genre'].nunique()}")
    
    print("\nAudio Features Statistics:")
    print(df[AUDIO_FEATURES].describe().round(2))
    
    print("\nRelease Date Range:")
    df_copy = df.copy()
    df_copy['Release Date'] = pd.to_datetime(df_copy['Release Date'], errors='coerce')
    print(f"  Earliest: {df_copy['Release Date'].min()}")
    print(f"  Latest: {df_copy['Release Date'].max()}")
    
    print("\nPopularity Stats:")
    print(f"  Mean: {df['popularity'].mean():.2f}")
    print(f"  Median: {df['popularity'].median():.2f}")
    print(f"  Std: {df['popularity'].std():.2f}")
    
    print("\nLength Stats (minutes):")
    length_min = pd.to_numeric(df['Length'], errors='coerce') / 60000
    print(f"  Mean: {length_min.mean():.2f}")
    print(f"  Median: {length_min.median():.2f}")
    
    print("\nContext Tags Coverage:")
    for col in CONTEXT_COLS:
        count = df[col].sum()
        print(f"  {col}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print("\n" + "="*60)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")
    
    setup_style()
    
    try:
        df = load_data()
        
        # Print summary first
        print_dataset_summary(df)
        
        # Original visualizations
        print("\n[*] Generating visualizations...")
        plot_emotion_distribution(df)
        plot_audio_features_boxplot(df)
        plot_audio_features_histogram(df)
        plot_correlation_matrix(df)
        plot_context_tags(df)
        plot_genre_distribution(df)
        
        # New visualizations
        plot_numeric_features(df)
        plot_categorical_features(df)
        plot_release_date_trends(df)
        plot_tempo_distribution(df)
        plot_energy_valence_scatter(df)
        plot_tempo_energy_scatter(df)
        plot_audio_by_emotion_violin(df)
        plot_popularity_by_genre(df)
        
        print(f"\n[SUCCESS] Hoan tat! Tat ca {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])} bieu do da duoc luu trong thu muc '{OUTPUT_DIR}'")
        
    except Exception as e:
        print(f"[ERROR] Co loi xay ra: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
