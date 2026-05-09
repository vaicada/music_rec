"""Debug Model 2 similarity values."""
import sys, numpy as np
sys.path.insert(0, '.')
import faiss
from audio_model.clip_audio_bridge import CLIPAudioBridge

bridge = CLIPAudioBridge()

# === Test 1: recommend_from_song ===
result = bridge.recommend_from_song('Stronger', 'Kanye West', top_k=5)
if result:
    print("=== recommend_from_song distances (after formula) ===")
    for r in result['recommendations']:
        print(f"  similarity={r['similarity']:.6f}  song={r['song']}")
else:
    print("Song not found, using index 0 directly")

# === Test 2: Raw FAISS inner product values ===
print()
print("=== Raw FAISS distances (before formula) ===")
v = bridge.faiss_index.reconstruct(0)
v = np.expand_dims(v, 0).astype(np.float32)
print("Vector norm (before normalize):", np.linalg.norm(v))
faiss.normalize_L2(v)
print("Vector norm (after normalize):", np.linalg.norm(v))
dists, idxs = bridge.faiss_index.search(v, 10)
print("Raw inner product values:", [round(d, 6) for d in dists[0].tolist()])
print("Formula result (dist+1)/2:", [round((d+1)/2, 4) for d in dists[0].tolist()])

# === Test 3: Check if index contains normalized vectors ===
print()
print("=== Checking stored vector norms in index ===")
norms = []
for i in range(0, min(100, bridge.faiss_index.ntotal)):
    v2 = bridge.faiss_index.reconstruct(i)
    norms.append(np.linalg.norm(v2))
print(f"Sample norms (first 100 vectors): min={min(norms):.4f} max={max(norms):.4f} mean={sum(norms)/len(norms):.4f}")
print("(Should all be ~1.0 if L2-normalized)")
