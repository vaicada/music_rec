"""
Visualize tracks_features.csv dataset (Model 2 - Audio Autoencoder)
Tạo 2 biểu đồ tối ưu nhất để mô tả dataset 1.2M bài hát thuần audio features.

Chart 1: Audio Feature Profile by Decade (Radar/Spider Chart)
  - Thể hiện sự tiến hóa trung bình của 8 audio features qua các thập kỷ (1960s-2020s)
  - Giúp hiểu cách âm nhạc thay đổi theo thời gian

Chart 2: Audio Features Pairwise Density (Hexbin Joint Grid)
  - Scatter-density map kết hợp Energy vs Danceability, Valence vs Acousticness
  - Thể hiện cấu trúc phân phối đặc trưng raw dataset Model 2 không có nhãn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

print("Loading tracks_features.csv (this may take a moment)...")
df = pd.read_csv('dataset/tracks_features.csv')
print(f"Dataset loaded: {len(df):,} tracks x {len(df.columns)} features")
print(f"Columns: {list(df.columns)}")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")

# ===========================================================================
# CHART 1: Audio Feature Radar by Decade
# ===========================================================================

print("\n[Chart 1] Creating Audio Feature Radar by Decade...")

# Normalize audio features to 0-1 scale
audio_features = ['danceability', 'energy', 'speechiness', 'acousticness',
                  'instrumentalness', 'liveness', 'valence']

# Normalize tempo and loudness to 0-1
df['tempo_norm'] = (df['tempo'] - df['tempo'].min()) / (df['tempo'].max() - df['tempo'].min())
df['loudness_norm'] = (df['loudness'] - df['loudness'].min()) / (df['loudness'].max() - df['loudness'].min())

features_normalized = audio_features + ['tempo_norm', 'loudness_norm']
feature_labels = ['Danceability', 'Energy', 'Speechiness', 'Acousticness',
                  'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Loudness']

# Assign decade
df['decade'] = (df['year'] // 10) * 10

# Filter decades with enough data, focus on 1960s-2020s
decade_counts = df[df['decade'] >= 1960]['decade'].value_counts().sort_index()
print(f"Decade counts:\n{decade_counts}")

target_decades = [1960, 1970, 1980, 1990, 2000, 2010, 2020]
df_filtered = df[df['decade'].isin(target_decades)]

# Compute mean per decade
decade_means = df_filtered.groupby('decade')[features_normalized].mean()
print(f"\nDecade means:\n{decade_means.round(3)}")

# Color palette per decade - cool to warm progression
decade_colors = {
    1960: '#6366F1',  # Indigo
    1970: '#8B5CF6',  # Purple
    1980: '#EC4899',  # Pink
    1990: '#EF4444',  # Red
    2000: '#F97316',  # Orange
    2010: '#EAB308',  # Yellow
    2020: '#22C55E',  # Green
}

# ---- Radar chart setup ----
N = len(features_normalized)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # Close the loop

fig1, ax1 = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True),
                          facecolor='#0F0F1A')
ax1.set_facecolor('#0F0F1A')

# Draw background rings
for r in np.arange(0.1, 1.1, 0.2):
    ax1.plot(angles, [r] * (N+1), color='white', alpha=0.08, linewidth=0.7, linestyle='--')
    ax1.fill(angles, [r] * (N+1), alpha=0.0)

# Draw spokes
for angle in angles[:-1]:
    ax1.plot([angle, angle], [0, 1], color='white', alpha=0.1, linewidth=0.7)

# Plot each decade
for decade in target_decades:
    if decade not in decade_means.index:
        continue
    values = decade_means.loc[decade, features_normalized].values.tolist()
    values += values[:1]  # close loop

    color = decade_colors[decade]
    ax1.plot(angles, values, 'o-', linewidth=2.5, label=f"{decade}s",
             color=color, markersize=6, alpha=0.95)
    ax1.fill(angles, values, alpha=0.08, color=color)

# Set axis labels
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(feature_labels, size=13, color='white', fontweight='bold', fontfamily='sans-serif')
ax1.set_ylim(0, 1)
ax1.set_yticks([0.2, 0.4, 0.6, 0.8])
ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], color='#AAAAAA', size=9)
ax1.tick_params(axis='x', pad=20)

# Remove default radial lines
ax1.spines['polar'].set_visible(False)
ax1.grid(False)

# Legend
legend = ax1.legend(
    loc='lower center', bbox_to_anchor=(0.5, -0.18),
    ncol=7, fontsize=13, frameon=True,
    facecolor='#1A1A2E', edgecolor='#444466', labelcolor='white',
    title='Decade', title_fontsize=13,
    markerscale=1.5
)

# Title
fig1.text(0.5, 0.97, "🎵 Audio Feature Profile Evolution by Decade", 
          ha='center', va='top', fontsize=20, fontweight='bold', 
          color='white', fontfamily='sans-serif')
fig1.text(0.5, 0.93, "tracks_features.csv  •  9 Audio Features normalized 0–1  •  1960s–2020s",
          ha='center', va='top', fontsize=12, color='#AAAACC', fontfamily='sans-serif')

# Dataset info box
info_text = f"Dataset: {len(df):,} tracks\nYears: {int(df['year'].min())}–{int(df['year'].max())}"
fig1.text(0.88, 0.06, info_text, ha='right', va='bottom', fontsize=10,
          color='#8888AA', fontfamily='monospace',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#1A1A2E', edgecolor='#444466', alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
out1 = 'visualizations/tracks_features_radar_by_decade.png'
fig1.savefig(out1, dpi=150, bbox_inches='tight', facecolor='#0F0F1A')
plt.close(fig1)
print(f"Chart 1 saved: {out1}")


# ===========================================================================
# CHART 2: 4-Panel Audio Feature Hexbin Density Matrix
# ===========================================================================

print("\n[Chart 2] Creating Audio Feature Hexbin Density Matrix...")

# Sample for performance (1.2M is too heavy for hexbin rendering)
sample_size = 200_000
df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
print(f"Using {len(df_sample):,} sample for density charts")

fig2 = plt.figure(figsize=(18, 14), facecolor='#0F0F1A')
fig2.suptitle("🎶 Audio Feature Pairwise Density Map — tracks_features.csv",
               fontsize=22, fontweight='bold', color='white', y=0.98)
fig2.text(0.5, 0.95, f"Hexbin density of {len(df_sample):,} sampled tracks  •  Darker = more tracks concentrated",
          ha='center', fontsize=13, color='#AAAACC')

gs = GridSpec(2, 3, figure=fig2, hspace=0.4, wspace=0.35,
              left=0.07, right=0.95, top=0.90, bottom=0.08)

pairs = [
    ('energy', 'danceability', 'Energy vs Danceability', '#6366F1', '#EC4899'),
    ('valence', 'acousticness', 'Valence vs Acousticness', '#22C55E', '#06B6D4'),
    ('energy', 'valence', 'Energy vs Valence', '#F97316', '#EAB308'),
    ('speechiness', 'instrumentalness', 'Speechiness vs Instrumentalness', '#8B5CF6', '#EC4899'),
    ('tempo_norm', 'energy', 'Tempo vs Energy', '#EF4444', '#F97316'),
    ('danceability', 'valence', 'Danceability vs Valence', '#06B6D4', '#22C55E'),
]

for idx, (x_col, y_col, title, c_start, c_end) in enumerate(pairs):
    row, col = divmod(idx, 3)
    ax = fig2.add_subplot(gs[row, col])
    ax.set_facecolor('#0F0F1A')

    # Custom colormap for this pair
    cmap = mcolors.LinearSegmentedColormap.from_list(
        f'custom_{idx}', ['#0F0F1A', c_start, c_end, 'white'], N=256
    )

    x_data = df_sample[x_col].dropna()
    y_data = df_sample[y_col].dropna()
    valid_mask = df_sample[[x_col, y_col]].dropna().index
    x_data = df_sample.loc[valid_mask, x_col]
    y_data = df_sample.loc[valid_mask, y_col]

    hb = ax.hexbin(x_data, y_data, gridsize=60, cmap=cmap, mincnt=1,
                   bins='log', linewidths=0.1)

    cb = fig2.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label('log(count)', color='#AAAACC', fontsize=9)
    cb.ax.yaxis.set_tick_params(color='#AAAACC', labelcolor='#AAAACC')

    # Compute and draw correlation
    corr = np.corrcoef(x_data, y_data)[0, 1]
    ax.text(0.97, 0.97, f'r = {corr:.3f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=11, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1A1A2E', alpha=0.8, edgecolor='#444466'))

    # Styling
    ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=10)
    x_label = x_col.replace('_norm', '').replace('_', ' ').capitalize()
    y_label = y_col.replace('_norm', '').replace('_', ' ').capitalize()
    ax.set_xlabel(x_label, color='#CCCCEE', fontsize=10)
    ax.set_ylabel(y_label, color='#CCCCEE', fontsize=10)
    ax.tick_params(colors='#888899', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

out2 = 'visualizations/tracks_features_hexbin_density.png'
fig2.savefig(out2, dpi=130, bbox_inches='tight', facecolor='#0F0F1A')
plt.close(fig2)
print(f"Chart 2 saved: {out2}")

print("\n✅ Both charts saved successfully!")
print(f"  1. {out1}")
print(f"  2. {out2}")
