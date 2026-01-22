# Vercel Deployment Checklist (LITE VERSION)

## ✅ Đã Hoàn Thành

- [x] Tạo system auto-download (download_helper.py)
- [x] Cấu hình .gitignore
- [x] **TỐI ƯU HÓA:** Không cần upload `train.csv` (1GB) nữa! Sử dụng `song_metadata.csv` (có sẵn).

## 🚀 Các Bước Cần Làm Ngay

### 1. Commit Code Mới Nhất

```bash
git add .
git commit -m "Deploy: Optimize for Vercel (Lite Version)"
git push origin main
```

### 2. Upload 3 Files Chinh lên Google Drive

Chỉ cần upload các file model (Tổng ~210MB thay vì 1.2GB):

- [ ] `models/best_model.pth` (~5 MB)
- [ ] `models/faiss_index.bin` (~135 MB)
- [ ] `models/faiss_index.bin.mappings.pkl` (~71 MB)

*(File `audio_stats.json` và `song_metadata.csv` ĐÃ CÓ trên GitHub, không cần upload)*

### 3. Cập Nhật File IDs

Mở `web_app/download_helper.py` và cập nhật IDs cho 3 file trên.

```python
FILE_CONFIGS = {
    "model": { ... "url": "..." },          # Update
    "faiss_index": { ... "url": "..." },    # Update
    "faiss_mappings": { ... "url": "..." }, # Update
    # "train_data": ... (ĐÃ COMMENT OUT - KHÔNG CẦN)
    "audio_stats": ... (Small file, can keep or ignore if on github)
}
```

### 4. Deploy lên Vercel

1. New Project → Import GitHub Repo
2. Build Command: `pip install -r requirements.txt && pip install -r web_app/requirements-deploy.txt`
3. Click **Deploy**

---

## � Tại sao không cần train.csv?

- `train.csv` (1GB) chứa lời bài hát và audio features chi tiết (dùng để train).
- `song_metadata.csv` (29MB) chỉ chứa tên bài, ca sĩ, genre (đủ để hiển thị kết quả).
- App sẽ tự động dùng `song_metadata.csv` nếu không thấy `train.csv`.
- **Lợi ích:** Tiết kiệm 1GB băng thông và bộ nhớ RAM cho Vercel!
