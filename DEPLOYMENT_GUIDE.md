# =============================================================================

# DEPLOYMENT GUIDE: Music Recommender Web App to Vercel

# =============================================================================

## 🎯 OVERVIEW

This guide explains how to deploy the Music Recommender Web App to Vercel using external file hosting for large model files.

**Problem:** GitHub and Vercel have file size limits:

- GitHub: 100MB per file
- Vercel: ~250MB total deployment size
- Our files: ~1.5GB total (model + index + data)

**Solution:** Host large files on Google Drive and auto-download on app startup.

---

## 📋 PREREQUISITES

1. **Google Drive account** (for hosting files)
2. **Vercel account** (free tier is enough)
3. **GitHub account** (for code repository)
4. **Local environment** with Python 3.11+

---

## 🚀 DEPLOYMENT STEPS

### STEP 1: Upload Files to Google Drive

Upload these files to your Google Drive (Total ~210 MB):

```
📦 Files to upload:
├── best_model.pth              (~5 MB)
├── faiss_index.bin             (~135 MB)
└── faiss_index.bin.mappings.pkl (~71 MB)
```

**Note:** We optimize the deployment by skipping `train.csv` (1GB). The app will automatically fall back to `models/song_metadata.csv` (29MB) which is already included in the GitHub repository.

**Location on your system:**

- `models/best_model.pth`
- `models/faiss_index.bin`
- `models/faiss_index.bin.mappings.pkl`

**How to upload:**

1. Go to [drive.google.com](https://drive.google.com)
2. Create a new folder (e.g., "music_recommender_models")
3. Upload the 3 files above
4. For EACH file:
   - Right-click → Share
   - Change to "Anyone with the link can view"
   - Copy the link

**Example link format:**

```
https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz123456/view?usp=sharing
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              This is your FILE_ID
```

---

### STEP 2: Update download_helper.py with File IDs

Open `web_app/download_helper.py` and replace the `YOUR_*_FILE_ID` placeholders with your actual Google Drive file IDs:

```python
FILE_CONFIGS = {
    "model": {
        "url": "https://drive.google.com/uc?id=1AbCdEfGhIjKlMnOpQrStUvWxYz123456",  # YOUR best_model.pth ID
        ...
    },
    ...
    # Note: train_data is commented out as we use the lite version
}
```

**⚠️ Important:** Replace IDs for model, faiss_index, and faiss_mappings!

---

### STEP 3: Add Large Files to .gitignore

These files should NOT be pushed to GitHub:

```bash
# Already in .gitignore (verify):
data/processed/*.csv
models/*.pth
models/*.bin
models/*.pkl
models/*.npy
```

---

### STEP 4: Test Locally (OPTIONAL but RECOMMENDED)

Before deploying, test the download helper:

```bash
cd web_app
python download_helper.py
```

This will:

- Check if files exist
- Download missing files from Google Drive
- Show progress bars
- Report success/failure

**Expected output:**

```
=============================================================
🔍 CHECKING REQUIRED FILES
=============================================================
❌ Trained PyTorch model weights: Missing
❌ FAISS similarity search index: Missing
...
📥 Downloading Trained PyTorch model weights...
   Target: ../models/best_model.pth
   ...
✅ Successfully downloaded Trained PyTorch model weights (5.0 MB)
...
=============================================================
✅ ALL FILES READY - Application can start
=============================================================
```

---

### STEP 5: Push to GitHub

```bash
# Add all changes
git add .

# Commit
git commit -m "Deploy: Add external file hosting for Vercel deployment

- Created download_helper.py for auto-downloading large files
- Modified app.py to check files on startup
- Added requirements-deploy.txt with gdown dependency
- Updated .gitignore to exclude large model files"

# Push to GitHub
git push origin main
```

---

### STEP 6: Deploy to Vercel

1. **Go to Vercel:** [vercel.com](https://vercel.com)
2. **Click "Add New Project"**
3. **Import Git Repository:**
   - Select your GitHub repository
   - Framework Preset: **Other** (not Python, we'll configure manually)

4. **Configure Build Settings:**

   ```
   Build Command: pip install -r requirements.txt && pip install -r web_app/requirements-deploy.txt
   Output Directory: web_app
   Install Command: pip install -r requirements.txt
   ```

5. **Environment Variables:** (Click "Add")

   ```
   PYTHON_VERSION=3.11
   ```

6. **Click "Deploy"**

---

### STEP 7: Wait for Deployment

⏱️ **First deployment takes 10-20 minutes** because:

1. Vercel builds the environment
2. Installs Python dependencies (~2 min)
3. **App starts and downloads files from Google Drive (~5-15 min)**
   - best_model.pth: ~5 MB
   - faiss_index.bin: ~135 MB  
   - mappings.pkl: ~71 MB
   - train.csv: ~964 MB ⏳ (this one takes longest)
   - audio_stats.json: < 1 MB

**Check deployment logs** to see download progress:

```
[0/4] Checking required files...
📥 Downloading Trained PyTorch model weights...
✅ Successfully downloaded (5.0 MB)
...
✅ ALL FILES READY - Application can start
[1/4] Initializing recommendation engine...
[2/4] Loading model...
[3/4] Loading FAISS index...
[4/4] Loading song data...
[OK] Music Recommender API is ready!
```

---

## 📊 EXPECTED DEPLOYMENT METRICS

| Metric | Value |
|--------|-------|
| **Build Time** | ~2-3 min |
| **Download Time** | ~5-15 min (varies by network) |
| **Total First Deploy** | ~10-20 min |
| **Cold Start** | ~30-60 sec (subsequent visits) |
| **Response Time** | ~50-200ms (after loaded) |

---

## 🐛 TROUBLESHOOTING

### ❌ "Download failed" errors

**Cause:** Google Drive file IDs are incorrect or files aren't public.

**Solution:**

1. Verify all file IDs in `download_helper.py`
2. Check that files are set to "Anyone with link can view"
3. Try downloading manually: `https://drive.google.com/uc?id=YOUR_FILE_ID`

---

### ❌ "Module 'gdown' not found"

**Cause:** `requirements-deploy.txt` not installed.

**Solution:** Add to Vercel Build Command:

```bash
pip install -r web_app/requirements-deploy.txt
```

---

### ❌ Deployment timeout

**Cause:** Downloads taking too long (>10 min).

**Solutions:**

1. **Option A:** Increase Vercel timeout (Pro plan required)
2. **Option B:** Use smaller dataset (~5K songs instead of 441K)
3. **Option C:** Use AWS S3 (faster downloads) instead of Google Drive

---

### ❌ Out of memory

**Cause:** Model + index + data ~2GB RAM, Vercel free tier has 1GB limit.

**Solutions:**

1. Upgrade to Vercel Pro ($20/mo) → 3GB RAM
2. Use smaller model/dataset
3. Deploy to Railway/Render (larger free tier)

---

## 🎉 SUCCESS CHECKLIST

- [ ] Files uploaded to Google Drive
- [ ] All 5 file IDs updated in `download_helper.py`
- [ ] Local test passes: `python web_app/download_helper.py`
- [ ] Code pushed to GitHub (without large files!)
- [ ] Vercel deployment created
- [ ] First deployment completed (10-20 min)
- [ ] App accessible at Vercel URL
- [ ] Search function works
- [ ] Mood/Context recommendations work
- [ ] YouTube player works

---

## 📝 NOTES

- **Subsequent deploys** are faster (~3-5 min) because files are cached
- **Cold starts** (no traffic for 10+ min) will re-download files
- **Consider** using persistent storage (Vercel KV, AWS S3) for production
- **Monitor** Google Drive quota (free tier: 15GB storage, unlimited bandwidth)

---

## 🔗 USEFUL LINKS

- Vercel Dashboard: <https://vercel.com/dashboard>
- Google Drive: <https://drive.google.com>
- GitHub Repo: <https://github.com/vaicada/music_rec>
- API Docs (after deploy): <https://your-app.vercel.app/docs>

---

**Author:** Graduation Project  
**Date:** 2026-01-19  
**Version:** 1.0
