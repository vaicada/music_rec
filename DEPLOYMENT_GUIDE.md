# =============================================================================

# DEPLOYMENT GUIDE: Music Recommender Web App to Hugging Face Spaces

# =============================================================================

## OVERVIEW

This guide explains how to deploy the Music Recommender Web App to Hugging Face Spaces using external file hosting for large model files.

**Platform:** Hugging Face Spaces
**SDK:** Docker
**Hardware:** CPU Basic (2 vCPU, 16GB RAM) - Free Tier is sufficient

---

## PREREQUISITES

1. **Google Drive account** (for hosting files)
2. **Hugging Face account**
3. **Local environment** with Python 3.11+
4. **Git** installed

---

## DEPLOYMENT STEPS

### STEP 1: Upload Files to Google Drive

Upload these files to your Google Drive (Total ~210 MB):

```
📦 Files to upload:
├── best_model.pth              (~5 MB)
├── faiss_index.bin             (~135 MB)
└── faiss_index.bin.mappings.pkl (~71 MB)
```

**Note:** The app skips `train.csv` (1GB) and uses the lighter `models/song_metadata.csv` included in the repo.

**How to upload:**

1. Go to [drive.google.com](https://drive.google.com)
2. Create a new folder (e.g., "music_recommender_models")
3. Upload the 3 files above
4. For EACH file:
   - Right-click → Share
   - Change to "Anyone with the link can view"
   - Copy the link and extract the FILE ID

---

### STEP 2: Update download_helper.py with File IDs

Open `web_app/download_helper.py` and replace the `YOUR_*_FILE_ID` placeholders with your actual Google Drive file IDs.

---

### STEP 3: Create Hugging Face Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. **Space name:** `music-recommender` (or your choice)
4. **License:** MIT or Apache 2.0
5. **Space SDK:** **Docker** (Important! Do NOT select Gradio or Streamlit)
6. Click **Create Space**

---

### STEP 4: Push Code to Space

You can push code directly via browser or use Git.

**Using Git (Recommended):**

```bash
# Clone the empty space (replace USER with your username)
git clone https://huggingface.co/spaces/USER/music-recommender
cd music-recommender

# Copy your project files into this directory
# Ensure Dockerfile is at the root
# Ensure web_app/, hybrid_music_engine/, models/ folders are present

# Add files
git add .
git commit -m "Initial commit"
git push
```

**Using Browser:**
Drag and drop your project files into the "Files" tab of your Space.

---

### STEP 5: Monitor Deployment

1. Go to the **App** tab of your Space.
2. You will see "Building" status.
3. Click "Logs" to watch the progress.
   - It will build the Docker image.
   - Then it will run `download_helper.py` to fetch models.
   - Finally, `uvicorn` will start the server.

**Expected Log Success:**

```
[OK] Music Recommender API is ready!
Application startup complete.
```

---

## IMPORTANT NOTES

- **Port:** The Dockerfile must expose port **7860**.
- **Permissions:** The container runs as non-root user (ID 1000). The `Dockerfile` handles this.
- **YouTube Fallback:** If YouTube searches fail due to IP blocking, the app will automatically return a direct link instead of an embed.

---

## TROUBLESHOOTING

- **Build Failed?** Check the Logs tab. Common errors are missing dependencies in `web_app/requirements-deploy.txt`.
- **Download Failed?** Verify Google Drive File IDs are public.
- **Runtime Error?** Ensure you are not trying to write to read-only directories. Only `/app` or `/tmp` are writable.

---

**Author:** Graduation Project
**Date:** 2026-01-26
**Version:** 2.0 (Hugging Face Edition)
