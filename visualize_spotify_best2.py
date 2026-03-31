"""
2 bieu do TOI UU NHAT de mo ta bo du lieu spotify_dataset.csv (Model 1):

CHART 1: spotify_emotion_audio_radar.png
  Radar Chart overlay profile am thanh trung binh cua MOI emotion.
  - 13 emotion, 7 audio features, normalized 0-1

CHART 2: spotify_dataset_overview.png
  Dashboard 2x2:
  Panel A: Emotion distribution (bar)
  Panel B: Audio features boxplot
  Panel C: Tempo distribution (histogram coolwarm)
  Panel D: Emotion space bubble (valence vs energy)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# Load data
# =========================================================
print("Loading data/processed/audio_train.csv ...")
df_train = pd.read_csv('data/processed/audio_train.csv')
df_val   = pd.read_csv('data/processed/audio_val.csv')
df = pd.concat([df_train, df_val], ignore_index=True)
print("  Loaded:", len(df), "rows x", len(df.columns), "cols")
print("  Columns:", list(df.columns))

EMOTION_COL = 'emotion'
AUDIO_FEATS = [c for c in ['energy','danceability','valence','acousticness',
                            'speechiness','liveness','instrumentalness','tempo']
               if c in df.columns]
print("Emotion column:", EMOTION_COL)
print("Audio features:", AUDIO_FEATS)

# Normalize tempo to 0-1
if 'tempo' in df.columns:
    tmin, tmax = df['tempo'].min(), df['tempo'].max()
    df['tempo_norm'] = (df['tempo'] - tmin) / (tmax - tmin)
    RADAR_FEATS  = [c for c in AUDIO_FEATS if c != 'tempo'] + ['tempo_norm']
    RADAR_LABELS = [c for c in AUDIO_FEATS if c != 'tempo'] + ['tempo']
else:
    RADAR_FEATS  = AUDIO_FEATS
    RADAR_LABELS = AUDIO_FEATS

emotion_counts = df[EMOTION_COL].value_counts()
print("\nEmotion counts:")
print(emotion_counts.to_string())

main_emotions = emotion_counts[emotion_counts >= 500].index.tolist()
print("\nMain emotions (>=500):", main_emotions)
df_main = df[df[EMOTION_COL].isin(main_emotions)].copy()

# =========================================================
# Color map per emotion
# =========================================================
EMOC = {
    'joy':        '#FBBF24',
    'sadness':    '#60A5FA',
    'anger':      '#F87171',
    'fear':       '#A78BFA',
    'love':       '#F472B6',
    'surprise':   '#34D399',
    'True':       '#94A3B8',
    'confusion':  '#F97316',
    'thirst':     '#06B6D4',
    'Love':       '#EC4899',
    'angry':      '#EF4444',
    'interest':   '#84CC16',
    'pink':       '#FB7185',
}

def ec(emo):
    return EMOC.get(emo, '#AAAACC')

# =========================================================
# CHART 1 — Emotion Audio Radar
# =========================================================
print("\n[Chart 1] Emotion Audio Radar...")

emotion_means = df_main.groupby(EMOTION_COL)[RADAR_FEATS].mean()
print("Emotion means:")
print(emotion_means.round(3))

N = len(RADAR_FEATS)
angles = (np.linspace(0, 2 * np.pi, N, endpoint=False)).tolist()
angles += angles[:1]

fig1, ax1 = plt.subplots(figsize=(13, 13), subplot_kw=dict(polar=True),
                          facecolor='#0A0A1A')
ax1.set_facecolor('#0A0A1A')

for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
    ax1.plot(angles, [r]*(N+1), color='white', alpha=0.07, linewidth=0.8, linestyle='--')
for angle in angles[:-1]:
    ax1.plot([angle, angle], [0, 1], color='white', alpha=0.08, linewidth=0.8)

legend_handles = []
for emo in emotion_means.index:
    vals = emotion_means.loc[emo, RADAR_FEATS].values.tolist()
    vals += vals[:1]
    color = ec(emo)
    ax1.plot(angles, vals, 'o-', lw=2.2, color=color, markersize=5, alpha=0.92)
    ax1.fill(angles, vals, alpha=0.07, color=color)
    cnt = int(emotion_counts.get(emo, 0))
    legend_handles.append(mpatches.Patch(color=color, label=str(emo) + '  (' + str(cnt) + ')'))

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels([l.replace('_', '\n').capitalize() for l in RADAR_LABELS],
                    size=13, color='white', fontweight='bold')
ax1.tick_params(axis='x', pad=22)
ax1.set_ylim(0, 1)
ax1.set_yticks([0.2, 0.4, 0.6, 0.8])
ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], color='#778899', size=9)
ax1.spines['polar'].set_visible(False)
ax1.grid(False)

ax1.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.22),
           ncol=4, fontsize=11.5, frameon=True, facecolor='#12122A',
           edgecolor='#334466', labelcolor='white', title='Emotion  (so bai hat)',
           title_fontsize=12, handlelength=1.4)

fig1.text(0.5, 0.97, "DNA Am Thanh Theo Cam Xuc  --  spotify_dataset.csv",
          ha='center', va='top', fontsize=20, fontweight='bold', color='white')
fig1.text(0.5, 0.93,
          "Radar chart - profile audio features trung binh cua moi emotion label  |  features chuan hoa 0-1",
          ha='center', va='top', fontsize=12, color='#99AACC')
info = "Dataset: " + str(len(df)) + " tracks  |  " + str(len(main_emotions)) + " emotions"
fig1.text(0.87, 0.05, info, ha='right', va='bottom', fontsize=10,
          color='#7788AA', fontfamily='monospace',
          bbox=dict(boxstyle='round,pad=0.4', fc='#12122A', ec='#334466', alpha=0.85))

plt.tight_layout(rect=[0, 0.08, 1, 0.92])
out1 = 'visualizations/spotify_emotion_audio_radar.png'
fig1.savefig(out1, dpi=150, bbox_inches='tight', facecolor='#0A0A1A')
plt.close(fig1)
print("Chart 1 saved:", out1)


# =========================================================
# CHART 2 — Dataset Overview Dashboard 2x2
# =========================================================
print("\n[Chart 2] Dataset Overview Dashboard...")

fig2 = plt.figure(figsize=(18, 13), facecolor='#0A0A1A')
fig2.suptitle("Tong Quan Bo Du Lieu spotify_dataset.csv",
               fontsize=22, fontweight='bold', color='white', y=0.98)
fig2.text(0.5, 0.94,
          str(len(df)) + " bai hat  |  " + str(df[EMOTION_COL].nunique()) +
          " emotion labels  |  du lieu Model 1 (Hybrid NLP + Audio)",
          ha='center', fontsize=13, color='#99AACC')

gs = GridSpec(2, 2, figure=fig2, hspace=0.42, wspace=0.32,
              left=0.07, right=0.96, top=0.88, bottom=0.07)

PANEL_BG   = '#11112A'
SPINE_CLR  = '#2A2A4A'
TICK_CLR   = '#778899'
TITLE_CLR  = '#DDEEFF'
TEXT_CLR   = '#CCDDEE'

# -- Panel A: Emotion Distribution ----------------------------------------
ax_a = fig2.add_subplot(gs[0, 0])
ax_a.set_facecolor(PANEL_BG)

ec_sorted   = emotion_counts.sort_values(ascending=False)
colors_bar  = [ec(e) for e in ec_sorted.index]
bars = ax_a.bar(range(len(ec_sorted)), ec_sorted.values,
                color=colors_bar, edgecolor='none', width=0.7, alpha=0.92)

for bar, v in zip(bars, ec_sorted.values):
    ax_a.text(bar.get_x() + bar.get_width()/2, v + max(ec_sorted.values)*0.01,
              str(v), ha='center', va='bottom', fontsize=7.5, color='white', fontweight='bold')

ax_a.set_xticks(range(len(ec_sorted)))
ax_a.set_xticklabels(list(ec_sorted.index), rotation=38, ha='right',
                     fontsize=9.5, color=TEXT_CLR)
ax_a.set_title("A  -  Phan Bo Emotion (Label Distribution)", color=TITLE_CLR,
               fontsize=13, fontweight='bold', pad=10)
ax_a.set_ylabel("So bai hat", color=TEXT_CLR, fontsize=10)
ax_a.tick_params(colors=TICK_CLR, labelsize=9)
for sp in ax_a.spines.values(): sp.set_edgecolor(SPINE_CLR)
ax_a.yaxis.grid(True, color=SPINE_CLR, linewidth=0.6, alpha=0.7)
ax_a.set_axisbelow(True)

top3_pct = ec_sorted.values[:3].sum() / ec_sorted.values.sum() * 100
ax_a.text(0.98, 0.97, "Top 3 = " + str(round(top3_pct,1)) + "% dataset",
          transform=ax_a.transAxes, ha='right', va='top', fontsize=9.5,
          color='#FFDD88', bbox=dict(boxstyle='round,pad=0.3', fc='#1A1A3A', ec='#665500', alpha=0.85))

# -- Panel B: Audio Features Boxplot --------------------------------------
ax_b = fig2.add_subplot(gs[0, 1])
ax_b.set_facecolor(PANEL_BG)

feat_01  = [f for f in ['danceability','energy','valence','acousticness',
                         'speechiness','liveness','instrumentalness'] if f in df.columns]
box_data = [df[f].dropna().values for f in feat_01]
bp = ax_b.boxplot(box_data, patch_artist=True, notch=False,
                  medianprops=dict(color='white', linewidth=2.2),
                  whiskerprops=dict(color='#8899BB', linewidth=1.2),
                  capprops=dict(color='#8899BB', linewidth=1.5),
                  flierprops=dict(marker='.', markerfacecolor='#445566',
                                  markersize=1.5, alpha=0.4))

bx_colors = ['#6366F1','#EC4899','#FBBF24','#22C55E','#F97316','#06B6D4','#A78BFA']
for patch, color in zip(bp['boxes'], bx_colors[:len(feat_01)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax_b.set_xticks(range(1, len(feat_01)+1))
ax_b.set_xticklabels([f.capitalize()[:11] for f in feat_01],
                     rotation=28, ha='right', fontsize=10, color=TEXT_CLR)
ax_b.set_title("B  -  Phan Phoi Audio Features (7 dac trung, 0-1)", color=TITLE_CLR,
               fontsize=13, fontweight='bold', pad=10)
ax_b.set_ylabel("Gia tri feature", color=TEXT_CLR, fontsize=10)
ax_b.tick_params(colors=TICK_CLR)
for sp in ax_b.spines.values(): sp.set_edgecolor(SPINE_CLR)
ax_b.yaxis.grid(True, color=SPINE_CLR, linewidth=0.6, alpha=0.7)
ax_b.set_axisbelow(True)

# -- Panel C: Tempo Distribution ------------------------------------------
ax_c = fig2.add_subplot(gs[1, 0])
ax_c.set_facecolor(PANEL_BG)

tempo_data = df['tempo'].dropna()
n_c, bins_c, patches_c = ax_c.hist(tempo_data, bins=60, edgecolor='none', alpha=0.88)
norm_c = mcolors.Normalize(vmin=bins_c.min(), vmax=bins_c.max())
cmap_c = plt.cm.coolwarm
for patch, left in zip(patches_c, bins_c[:-1]):
    patch.set_facecolor(cmap_c(norm_c(left)))

mean_t = tempo_data.mean()
med_t  = tempo_data.median()
ax_c.axvline(mean_t, color='#FBBF24', lw=2.0, linestyle='--',
             label='Mean = ' + str(round(mean_t, 1)) + ' BPM')
ax_c.axvline(med_t,  color='#FFFFFF', lw=1.5, linestyle=':',
             label='Median = ' + str(round(med_t, 1)) + ' BPM')
ax_c.legend(fontsize=9.5, facecolor='#1A1A3A', edgecolor='#334466', labelcolor='white')
ax_c.set_title("C  -  Phan Phoi Tempo (BPM)  --  Nhip do bai hat", color=TITLE_CLR,
               fontsize=13, fontweight='bold', pad=10)
ax_c.set_xlabel("Tempo (BPM)", color=TEXT_CLR, fontsize=10)
ax_c.set_ylabel("So bai hat", color=TEXT_CLR, fontsize=10)
ax_c.tick_params(colors=TICK_CLR, labelsize=9)
for sp in ax_c.spines.values(): sp.set_edgecolor(SPINE_CLR)
ax_c.yaxis.grid(True, color=SPINE_CLR, linewidth=0.6, alpha=0.7)
ax_c.set_axisbelow(True)

# -- Panel D: Emotion Bubble (Valence vs Energy) --------------------------
ax_d = fig2.add_subplot(gs[1, 1])
ax_d.set_facecolor(PANEL_BG)

em_stats = df.groupby(EMOTION_COL).agg(
    energy_mean  = ('energy',  'mean'),
    valence_mean = ('valence', 'mean'),
    cnt          = (EMOTION_COL, 'size')
).reset_index()

max_cnt = em_stats['cnt'].max()
for _, row in em_stats.iterrows():
    color    = ec(row[EMOTION_COL])
    sz       = row['cnt'] / max_cnt * 2000 + 80
    ax_d.scatter(row['valence_mean'], row['energy_mean'],
                 s=sz, color=color, alpha=0.82,
                 edgecolors='white', linewidths=0.7, zorder=3)
    ax_d.annotate(str(row[EMOTION_COL]),
                  (row['valence_mean'], row['energy_mean']),
                  fontsize=8.5, color='white', ha='center', va='bottom',
                  xytext=(0, 9), textcoords='offset points',
                  bbox=dict(boxstyle='round,pad=0.2', fc='#12122A', ec='none', alpha=0.7))

ax_d.axvline(0.5, color='white', lw=0.8, linestyle='--', alpha=0.25)
ax_d.axhline(0.5, color='white', lw=0.8, linestyle='--', alpha=0.25)
ax_d.text(0.02, 0.97, 'Quiet\n& Sad',      transform=ax_d.transAxes, color='#99AACC', fontsize=8, va='top')
ax_d.text(0.72, 0.97, 'Happy\n& Energetic', transform=ax_d.transAxes, color='#99AACC', fontsize=8, va='top')
ax_d.text(0.02, 0.02, 'Calm\n& Mellow',    transform=ax_d.transAxes, color='#99AACC', fontsize=8, va='bottom')
ax_d.text(0.72, 0.02, 'Dark\n& Intense',   transform=ax_d.transAxes, color='#99AACC', fontsize=8, va='bottom')

ax_d.set_xlim(-0.05, 1.05)
ax_d.set_ylim(-0.05, 1.05)
ax_d.set_title("D  -  Khong Gian Cam Xuc: Valence vs Energy", color=TITLE_CLR,
               fontsize=13, fontweight='bold', pad=10)
ax_d.set_xlabel("Mean Valence (tich cuc <-> tieu cuc)", color=TEXT_CLR, fontsize=10)
ax_d.set_ylabel("Mean Energy (nang luong)", color=TEXT_CLR, fontsize=10)
ax_d.tick_params(colors=TICK_CLR, labelsize=9)
for sp in ax_d.spines.values(): sp.set_edgecolor(SPINE_CLR)
ax_d.grid(True, color=SPINE_CLR, linewidth=0.5, alpha=0.5)
ax_d.set_axisbelow(True)

out2 = 'visualizations/spotify_dataset_overview.png'
fig2.savefig(out2, dpi=140, bbox_inches='tight', facecolor='#0A0A1A')
plt.close(fig2)
print("Chart 2 saved:", out2)

print("\nDone!")
print("  1.", out1)
print("  2.", out2)
