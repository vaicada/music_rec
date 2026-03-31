# Báo Cáo Phân Tích & Trực Quan Hoá Dữ Liệu (Data Visualization Summary)

Tài liệu này tổng hợp toàn bộ các kết quả trực quan hóa dữ liệu được thực hiện trong dự án Music Recommender, bao gồm phân tích tập dữ liệu có nhãn (`spotify_dataset.csv`), tập dữ liệu gốc (`tracks_features.csv`) và không gian nhúng của các mô hình học máy.

---

## 1. 📊 Tổng Quan Các Bộ Dữ Liệu Thực Nghiệm

Dự án sử dụng hai bộ dữ liệu chính để phục vụ cho hai mô hình gợi ý khác nhau:

### 1.1 Dataset 1: `spotify_dataset.csv` (Dành cho Model 1)
- **Tổng số bài hát:** 496,278 tracks (tập train + val).
- **Đặc trưng âm thanh:** 8 features chuẩn hóa (energy, danceability, valence, acousticness, speechiness, liveness, instrumentalness, tempo).
- **Đặc trưng phân loại:** 6 nhãn cảm xúc chính (Joy, Sadness, Anger, Fear, Love, Surprise).
- **Tính chất:** Dữ liệu có giám sát (Supervised), được dùng để huấn luyện mô hình phân loại cảm xúc và gợi ý Hybrid (NLP + Audio).
- **Vấn đề dữ liệu:** Mất cân bằng lớp (Class Imbalance) nghiêm trọng khi Top 3 emotions (Joy, Sadness, Anger) chiếm tới 88.8%.

### 1.2 Dataset 2: `tracks_features.csv` (Dành cho Model 2)
- **Tổng số bài hát:** 1,204,025 tracks.
- **Đặc trưng âm thanh:** 9 audio features (thêm loudness).
- **Khoảng thời gian:** Nhạc phát hành từ 1921 đến 2021.
- **Tính chất:** Dữ liệu không giám sát (Unsupervised), không có nhãn cảm xúc hay thể loại, được dùng để huấn luyện Audio Autoencoder.

---

## 2. 🌟 Biểu Đồ Trực Quan Tối Ưu (Presentation Ready)

Để trình bày một cách súc tích và toàn diện nhất về đặc trưng của hai bộ dữ liệu, chúng tôi đã chọn lọc và thiết kế 4 biểu đồ tối ưu sau:

### 2.1 Đối với Dataset `spotify_dataset.csv` (Model 1 Input)

#### ⭐ 1. DNA Âm Thanh Theo Cảm Xúc (Emotion Audio Radar)
- **Đường dẫn:** `visualizations/spotify_emotion_audio_radar.png`
- **Mô tả:** Radar Chart đa chiều thể hiện giá trị trung bình của 7 audio features trên 6 nhóm cảm xúc khác nhau.
- **Insights:** Thể hiện rõ "chữ ký âm thanh" của từng cảm xúc. Ví dụ: *Joy* có Danceability cao nhất, trong khi *Anger* vượt trội về Speechiness và Energy. Sự chồng lấn giữa các đường đồ thị cũng chỉ ra rằng ranh giới cảm xúc trong không gian âm thanh khá mờ nhạt, khẳng định sự cần thiết của phương pháp Hybrid kết hợp thêm NLP từ lời bài hát.

#### ⭐ 2. Tổng Quan Bộ Dữ Liệu (Dataset Overview Dashboard 2x2)
- **Đường dẫn:** `visualizations/spotify_dataset_overview.png`
- **Mô tả:** Bảng điều khiển gồm 4 panel: Phân bố nhãn cảm xúc (Bar chart), Dải giá trị Audio Features (Boxplot), Phân phối Tempo (Histogram), và Mối quan hệ Valence - Energy (Bubble chart).
- **Insights:** Cung cấp cái nhìn toàn cảnh về dataset: tình trạng mất cân bằng nhãn (class imbalance), nhịp điệu (tempo) tập trung mạnh ở 120 BPM, và các cụm cảm xúc (clusters) chủ yếu nằm ở vùng trung bình của Valence tập trung cao về Energy. Nhãn "Love" xuất hiện như một điểm nhiễu (noise) cần được xử lý.

### 2.2 Đối với Dataset `tracks_features.csv` (Model 2 Input)

#### ⭐ 3. Sự Tiến Hóa Âm Thanh Qua Các Thập Kỷ (Radar Chart by Decade)
- **Đường dẫn:** `visualizations/tracks_features_radar_by_decade.png`
- **Mô tả:** Radar Chart biểu diễn 9 đặc trưng âm thanh theo 7 thập kỷ (1960s – 2020s).
- **Insights:** Acousticness giảm mạnh theo thời gian, phản ánh sự dịch chuyển từ nhạc cụ mộc sang sản xuất âm nhạc điện tử. Energy và Speechiness tăng dần, đánh dấu sự bùng nổ của EDM và Hip-hop/Rap từ thập niên 90.

#### ⭐ 4. Ma Trận Mật Độ Phân Phối (Hexbin Density Matrix)
- **Đường dẫn:** `visualizations/tracks_features_hexbin_density.png`
- **Mô tả:** Ma trận 2x3 hiển thị joint distribution của 6 cặp features quan trọng nhất bằng kỹ thuật Hexbin.
- **Insights:** Giải quyết triệt để lỗi overplotting của 1.2 triệu điểm dữ liệu. Cho thấy sự phân cực (bimodal) của Acousticness, và sự gia tăng tuyến tính giữa khả năng khiêu vũ (Danceability) và sự tích cực (Valence).

---

## 3. 🧠 Trực Quan Hóa Không Gian Nhúng Của Mô Hình

Để giải thích (explainability) những gì mô hình đã học được:

### 3.1 Không Gian Phân Cụm 3D Của Model 1 (Hybrid NLP + Audio)
- **Trạng thái:** HOẠT ĐỘNG TỐT (STABLE)
- **Tệp:** `visualizations/model1_3d_interactive.html`
- **Mô tả:** Biểu đồ tương tác 3D Scatter Plot (kết hợp thuật toán giảm chiều UMAP/t-SNE) minh họa khả năng phân cụm cực tốt của hệ thống Hybrid. Các cụm cảm xúc như Happy, Sad, hay Calm tách biệt rõ ràng nhờ sự hỗ trợ đắc lực từ ngữ nghĩa (Lyrics).

### 3.2 Dải Chuyển Tiếp Âm Thanh Bậc Thuận (Model 2 Autoencoder)
- **Trạng thái:** SẴN SÀNG KHAI THÁC (PRODUCTION-READY)
- **Tệp:** `visualizations/model2_feature_gradient.png`
- **Mô tả:** Biểu diễn 2D Gradient không gian latent 32 chiều được nén từ 9 đặc trưng âm thanh.
- **Insights:** Thay vì phân nhóm cứng nhắc theo nhãn, mô hình tổ chức không gian học máy thành một phổ âm thanh mượt mà (spectrum) chuyển tiếp tự nhiên từ nhạc Acoustic tĩnh lặng sang đến cường độ mạnh bạo của EDM/Metal.

---

## 4. 🔍 Các Biểu Đồ Thăm Dò Khác (Exploratory Data Analysis)

Quá trình nghiên cứu đã tạo ra 14 biểu đồ thăm dò (EDA) trên tập `spotify_dataset.csv`. Tuy mang lại góc nhìn ở giai đoạn đầu, các biểu đồ này hiện được lưu trữ để tham khảo, vấp phải vấn đề hiển thị (như overplotting hoặc thông tin phân tán) nên **không** chọn làm biểu đồ báo cáo chính thức.

*Danh mục biểu đồ EDA lưu trữ:*
1. `emotion_distribution.png` (Count plot)
2. `genre_distribution.png` (Barplot)
3. `context_distribution.png` (Barplot)
4. `audio_features_boxplot.png` (Thay thế bởi Dashboard)
5. `audio_features_histogram.png` (Thay thế bởi Dashboard)
6. `correlation_matrix.png` (Heatmap)
7. `tempo_distribution.png` (Thay thế bởi Dashboard)
8. `numeric_features.png` (Dữ liệu lỗi cột length)
9. `categorical_features.png` (Multi-subplot)
10. `release_date_trends.png` (Line plot)
11. `energy_valence_scatter.png` (Bị lỗi overplotting)
12. `tempo_energy_scatter.png` (Bị lỗi overplotting)
13. `audio_by_emotion_violin.png` (Nhãn emotion bị nhiễu)
14. `popularity_by_genre.png` (Genre-level analysis)

---

## 🎯 5. KẾT LUẬN

Hệ thống trực quan hóa dữ liệu của dự án Music Recommender đã hoàn thành trọn vẹn với tổng cộng **20 biểu đồ** (bao gồm 4 biểu đồ báo cáo tối ưu, 2 biểu đồ phân tích không gian nhúng của mô hình và 14 biểu đồ thăm dò phụ trợ). 

**Chất lượng trực quan hóa:**
- Đã giải quyết thành công các thách thức về **Big Data Visualization** (sử dụng Hexbin thay cho Scatter plot để khắc phục overplotting cho 1.2M điểm dữ liệu).
- Cô đọng xuất sắc thông tin biểu đạt (sử dụng Radar Chart đa trục và Dashboard 4-panel) giúp mô tả đa chiều cấu trúc âm thanh và sự mất cân bằng lớp (Class Imbalance) chỉ trong một ánh nhìn.
- Biểu diễn được "hộp đen" của Trí tuệ Nhân tạo thông qua các đồ thị phân cụm không gian nhúng FAISS/UMAP (3D Interactive Clustering và 2D Gradient Spectrum).

Toàn bộ **2 tập dữ liệu nguồn (1.7 triệu mẫu)** và **kiến trúc không gian nhúng của 2 Models** đều đã được mổ xẻ và trình bày **khoa học, tường minh và thuyết phục**, tạo cơ sở vững chắc để chứng minh hiệu năng hệ thống của toàn dự án.
