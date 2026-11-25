# Ứng Dụng Học Máy Trong Dự Đoán Tương Tác Thuốc - Thuốc và Khắc Phục Hiện Tượng Rò Rỉ Dữ Liệu

> **Học phần:** Học máy (Machine Learning) -- Học kỳ 1, Năm học 2026--2027  
> **Khoa Công nghệ Thông tin - Trường Đại học Công nghệ Thông tin và Truyền thông (ICTU)**  
> **Giảng viên hướng dẫn:** ThS. Nguyễn Tuấn Anh  
> **Sinh viên thực hiện:**  
> 1. Trần Đặng Vương Quốc Long - MSV: DTC235200472 (Lớp: KHMT K22A)  
> 2. Bùi Minh Ảnh - MSV: DTC235160004 (Lớp: KHMT K22)

---

## 📌 1. Giới thiệu bài toán
Dự đoán tương tác thuốc--thuốc (Drug-Drug Interaction - DDI) đóng vai trò then chốt trong y học nhằm cảnh báo sớm các phản ứng bất lợi khi kê đơn phối hợp thuốc. Đề tài tập trung giải quyết 2 thách thức lớn:
1. **Hiện tượng khởi động lạnh (Cold-start):** Dự đoán tương tác cho các loại thuốc mới chưa từng có liên kết trên mạng lưới.
2. **Hiện tượng rò rỉ dữ liệu cấu trúc (Data Leakage):** Thay thế cách chia ngẫu nhiên trên cạnh (Warm-start) bằng khung đánh giá 4 kịch bản phân tách thực thể thuốc rời rạc (Drug-disjoint).

---

## 🚀 2. Quy trình kỹ thuật (Pipeline)
1. **Khảo sát & Tiền xử lý (EDA):** Thống kê 548 loại thuốc, 48.584 liên kết DDI, xử lý khuyết thiếu và cân bằng mẫu bằng *Balanced Negative Sampling 1:1*.
2. **Lọc nhiễu Shannon Entropy ($\gamma \ge 0.85$):** Tinh chỉnh và loại bỏ các đặc trưng thưa/kém thông tin từ 10 nguồn tương đồng sinh học.
3. **Hợp nhất mạng tương đồng (Similarity Network Fusion - SNF):** Tích hợp thông tin đa nguồn (Hóa học, Đích sinh học, Enzyme, Chỉ định lâm sàng...) với tham số chuẩn $K=20, t=20$.
4. **Phân chia 4 kịch bản chống rò rỉ:**
   - **Case 1 (Warm-start):** Chia ngẫu nhiên theo cặp liên kết.
   - **Case 2 (Strict Cold-start):** Cả hai thuốc đều mới ($D_{\text{train}} \cap D_{\text{test}} = \emptyset$).
   - **Case 3 (Semi Cold-start):** Một thuốc cũ kết hợp một thuốc mới.
   - **Case 4 (Mixed Scenario):** Tổng hợp cả Case 2 và Case 3.
5. **Mô hình Soft Voting Ensemble:** Kết hợp xác suất dự đoán của 4 mô hình cơ sở (Random Forest, Logistic Regression, KNN, XGBoost) qua cơ chế trung bình cộng.

---

## 📊 3. Kết quả thực nghiệm
| Kịch bản đánh giá | Mô hình tốt nhất | ROC-AUC | PR-AUC (AUPR) | F1-score |
|---|---|:---:|:---:|:---:|
| **Case 1: Warm-start** | Logistic Regression / Ensemble | 0.9383 | 0.9441 | 0.8591 |
| **Case 2: Strict Cold-start** (New × New) | Soft Voting Ensemble | **0.6828** | **0.6791** | **0.6064** |
| **Case 3: Semi Cold-start** (Old × New) | Soft Voting Ensemble | **0.8428** | **0.8313** | **0.7521** |
| **Case 4: Mixed Scenario** | Soft Voting Ensemble | **0.8302** | **0.8202** | **0.7375** |

---

## 🛠️ 4. Hướng dẫn chạy thử nghiệm
```bash
# 1. Clone repository
git clone https://github.com/<tai-khoan-cua-ban>/DDI-MachineLearning-Project.git
cd DDI-MachineLearning-Project

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Chạy toàn bộ pipeline huấn luyện và đánh giá
python src/pipeline_end_to_end_ddi.py
```
