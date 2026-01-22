---
description: Quy tắc báo cáo sau mỗi tiến trình quan trọng trong dự án Music Recommender
---

# Quy Tắc Ghi Báo Cáo Dự Án

Sau mỗi tiến trình quan trọng (training, data preparation, evaluation, etc.), BẮT BUỘC phải ghi báo cáo chi tiết vào file tương ứng.

## Các Tiến Trình Cần Báo Cáo

1. **Data Preparation** → `DATA_PREPARATION_REPORT.md`
2. **Model Training** → `TRAINING_REPORT.md`
3. **FAISS Index Building** → `FAISS_INDEX_REPORT.md`
4. **Model Evaluation** → `EVALUATION_REPORT.md`
5. **Deployment** → `DEPLOYMENT_REPORT.md`

## Cấu Trúc Báo Cáo Chuẩn

Mỗi báo cáo PHẢI có các phần sau:

```markdown
# TIÊU ĐỀ BÁO CÁO

**Ngày thực hiện:** YYYY-MM-DD
**Thời gian xử lý:** X giờ Y phút
**Thiết bị:** (GPU/CPU info)

---

## 1. TỔNG QUAN
- Mục tiêu của tiến trình
- Input data/files
- Expected output

## 2. CÁC BƯỚC THỰC HIỆN CHI TIẾT
- Mô tả từng bước
- Code samples quan trọng (có thể copy vào báo cáo dự án)
- Giải thích TẠI SAO làm như vậy

## 3. VẤN ĐỀ GẶP PHẢI VÀ GIẢI PHÁP
- Lỗi/vấn đề đã gặp
- Cách giải quyết
- Lý do chọn giải pháp đó

## 4. KẾT QUẢ
- Thống kê định lượng (bảng số liệu)
- Files đã tạo (tên, kích thước)
- Metrics quan trọng

## 5. KIẾN TRÚC/SƠ ĐỒ (nếu có)
- Diagram ASCII hoặc mô tả
- Flow xử lý

## 6. LƯU Ý KỸ THUẬT
- Compatibility issues
- Performance notes
- Best practices

## 7. BƯỚC TIẾP THEO
- Các task cần làm sau tiến trình này
```

## Yêu Cầu Về Style

1. **Chi tiết:** Đủ thông tin để người khác hiểu được
2. **Chính xác:** Số liệu, metrics phải đúng
3. **Dễ hiểu:** Dùng tiếng Việt, giải thích thuật ngữ kỹ thuật
4. **Dễ trình bày:** Có bảng, code blocks, bullet points
5. **Copy-ready:** Code samples có thể copy trực tiếp vào báo cáo dự án

## Ví Dụ Files Báo Cáo Hiện Có

- `DATA_PREPARATION_REPORT.md` - Báo cáo chuẩn bị dữ liệu
- `PROJECT_WORK_LOG.md` - Log công việc tổng quát
- `EXECUTION_LOG.md` - Log thực thi chi tiết

## Lưu Ý

- Luôn tạo/cập nhật báo cáo NGAY SAU KHI tiến trình hoàn thành
- Không để thiếu bất kỳ bước nào đã thực hiện
- Ghi cả những thất bại/lỗi để rút kinh nghiệm
