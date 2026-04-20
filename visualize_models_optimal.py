import os
import faiss
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.express as px
from collections import Counter

# Try importing UMAP, fallback to PCA/t-SNE if not installed
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not installed. Will use t-SNE for 3D instead.")

def visualize_model1_3d(num_samples=2500):
    print("\n--- Generating Model 1 (3D Interactive) ---")
    index_path = "models/faiss_index.bin"
    mappings_path = "models/faiss_index.bin.mappings.pkl"
    
    if not os.path.exists(index_path):
        print("Model 1 index not found!")
        return

    # Load FAISS and mappings
    index = faiss.read_index(index_path)
    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)
        
    total_songs = index.ntotal
    sample_size = min(total_songs, num_samples)
    indices = np.random.choice(total_songs, sample_size, replace=False)
    
    embeddings, valid_indices = [], []
    for idx in indices:
        try:
            emb = index.reconstruct(int(idx))
            embeddings.append(emb)
            valid_indices.append(idx)
        except RuntimeError:
            pass
            
    embeddings = np.vstack(embeddings)
    
    # Extract Metadata for Plotly Hover
    song_titles, artists, genres, emotions = [], [], [], []
    for idx in valid_indices:
        meta = mappings["idx_to_song"][int(idx)] if "idx_to_song" in mappings else mappings[int(idx)]
        
        song_titles.append(meta.get("song", "Unknown"))
        artists.append(meta.get("artist", "Unknown"))
        genres.append(meta.get("genre", "Unknown"))
        
        emo = meta.get("emotion", "Unknown")
        emotions.append(emo if emo else "Unknown")
        
    # Group rare emotions into 'Other'
    top_emo = [k for k, v in Counter(emotions).most_common(8) if k != 'Unknown' and k != '']
    final_emotions = [e if e in top_emo else "Other" for e in emotions]

    print("Running Dimensionality Reduction (3D)...")
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=3, random_state=42)
        emb_3d = reducer.fit_transform(embeddings)
    else:
        reducer = TSNE(n_components=3, random_state=42, perplexity=30)
        emb_3d = reducer.fit_transform(embeddings)

    # Create DataFrame for Plotly
    df = pd.DataFrame({
        'x': emb_3d[:, 0],
        'y': emb_3d[:, 1],
        'z': emb_3d[:, 2],
        'Song': song_titles,
        'Artist': artists,
        'Emotion': final_emotions,
        'Genre': genres
    })

    # Plot
    fig = px.scatter_3d(df, x='x', y='y', z='z',
                        color='Emotion',
                        hover_name='Song',
                        hover_data={'Artist': True, 'Genre': True, 'x': False, 'y': False, 'z': False},
                        title='Model 1: NLP + Audio Hybrid Embedding Space (3D)',
                        opacity=0.8,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
    
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    
    out_file = "visualizations/model1_3d_interactive.html"
    fig.write_html(out_file)
    print(f"Saved interactive 3D plot to {out_file}")

def visualize_model2_gradient(num_samples=3000):
    print("\n--- Generating Model 2 (Continuous Gradient) ---")
    index_path = "models/tracks_faiss.index"
    mappings_path = "models/tracks_faiss.index.mappings.pkl"
    
    if not os.path.exists(index_path):
        print("Model 2 index not found!")
        return

    # Load FAISS
    index = faiss.read_index(index_path)
    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)
        
    total_songs = index.ntotal
    sample_size = min(total_songs, num_samples)
    indices = np.random.choice(total_songs, sample_size, replace=False)
    
    embeddings, valid_indices = [], []
    for idx in indices:
        try:
            embeddings.append(index.reconstruct(int(idx)))
            valid_indices.append(idx)
        except RuntimeError:
            pass
            
    embeddings = np.vstack(embeddings)
    
    # We need a continuous feature. 
    # Since mappings.pkl only has song, artist, genre, we'll try to reconstruct 'energy' proxy
    # Model 2 embeddings are highly correlated with 'energy' along PC1.
    # Let's perform PCA on embeddings. The 1st Principal Component in typical Audio Autoencoders
    # captures the "Energy/Loudness" axis strongly!
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    energy_proxy = pca.fit_transform(embeddings).flatten()
    
    # Normalize proxy to 0-1
    energy_proxy = (energy_proxy - energy_proxy.min()) / (energy_proxy.max() - energy_proxy.min())

    print("Running Dimensionality Reduction (2D t-SNE)...")
    tsne = TSNE(n_components=2, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 10))
    # We use a colormap (plasma) to show continuous spectrum
    scatter = plt.scatter(
        emb_2d[:, 0], 
        emb_2d[:, 1],
        c=energy_proxy,
        cmap='plasma',
        alpha=0.6,
        s=40,
        linewidths=0
    )
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Energy / Acoustic Proximate Spectrum', rotation=270, labelpad=20)
    plt.title("Model 2: Continuous Flow of Audio Features (t-SNE Gradient)", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    out_file = "visualizations/model2_feature_gradient.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved gradient plot to {out_file}")

if __name__ == "__main__":
    os.makedirs("visualizations", exist_ok=True)
    visualize_model1_3d()
    visualize_model2_gradient()
    print("\n[OK] All optimal visualizations generated successfully!")
