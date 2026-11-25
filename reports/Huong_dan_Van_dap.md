# HƯỚNG DẪN BẢO VỆ VẤN ĐÁP & THỰC HÀNH TẠI CHỖ -- BÀI KIỂM TRA 1 (TUẦN 4)
**Học phần:** Học máy (Machine Learning) -- Học kỳ 1, Năm học 2026--2027  
**Giảng viên:** Nguyễn Tuấn Anh  
**Sinh viên thực hiện:** 
1. Trần Đặng Vương Quốc Long -- **MSSV:** DTC235200472 -- **Lớp:** KHMT K22A  
2. Bùi Minh Ảnh -- **MSSV:** DTC235160004 -- **Lớp:** KHMT K22  
**Đề tài:** Ứng dụng học máy trong dự đoán tương tác thuốc--thuốc và khắc phục hiện tượng rò rỉ dữ liệu  

---

## I. TỔNG HỢP DANH MỤC FILE NỘP CHO BÀI KIỂM TRA 1

1. **Báo cáo tiểu luận:** `Bao_cao_Kiem_tra_1_Hoc_may.tex` (Được viết theo chuẩn mẫu `Mau bao cao mon Hoc may.tex` của bộ môn, đã biên soạn đầy đủ 3 chương, trang bìa, lời cam đoan, bảng phân công, từ điển dữ liệu, phân tích EDA, định lượng dữ liệu và phụ lục nhật ký tuần 1--4).
2. **Notebook thực nghiệm & thực hành:** `Kiem_tra_1_EDA_va_Tien_xu_ly.ipynb` (Đã chạy hoàn chỉnh toàn bộ output, tích hợp đầy đủ 10 tiêu chí và 3 hàm tương tác tại chỗ cho giảng viên kiểm tra).
3. **Mã nguồn tự động sinh biểu đồ:** `generate_eda_and_figures.py` (Kèm toàn bộ 7 biểu đồ khoa học phân giải cao trong thư mục `images/`).
4. **Báo cáo số liệu thống kê JSON:** `eda_statistics_report.json` (Chứa số liệu định lượng chính xác 100% của 548 thuốc, 48.584 tương tác DDI, 10 ma trận tương quan và tỷ lệ khuyết thiếu).

---

## II. BỘ CÂU HỎI VẤN ĐÁP & CÂU TRẢ LỜI MẪU (CHUẨN BỊ CHO TIÊU CHÍ [N] VÀ [CN])

> **Lưu ý quy chế chấm:** Theo Mục V của Tiêu chí chấm, giảng viên sẽ chọn ngẫu nhiên 2--3 tiêu chí [N] và 2 tiêu chí [CN] (Tiêu chí 9, 10) để hỏi nhanh sinh viên. Điểm cá nhân = MIN(Điểm nhóm, Mức hiểu cá nhân). Nắm chắc các câu trả lời dưới đây để đạt điểm tối đa 10/10!

---

### [N] Tiêu chí 1: Phát biểu bài toán học máy
* **Câu hỏi GV:** Bài toán của em thuộc loại bài toán học máy nào? Đơn vị mẫu, đầu vào và đầu ra là gì?
* **Trả lời mẫu:**
  - *Loại bài toán:* Học có giám sát (Supervised Learning), cụ thể là bài toán Phân loại nhị phân (Binary Classification) hoặc Dự đoán liên kết trên đồ thị (Link Prediction on Graph).
  - *Đơn vị mẫu:* Cặp thuốc vô hướng $(d_i, d_j)$ với $i < j$ từ tập 548 thuốc.
  - *Đầu vào:* Vector đặc trưng biểu diễn quan hệ giữa hai thuốc từ 10 kênh tương đồng sinh y học (SMILES cấu trúc hóa học, Protein đích, Enzyme chuyển hóa, Protein vận chuyển Transporter, Con đường sinh học Pathway, Chỉ định điều trị Indication, Tác dụng phụ Side effect, Biến cố bất lợi Offside effect, Mã ATC, Mã MeSH).
  - *Đầu ra:* Nhãn nhị phân $y_{ij} \in \{0, 1\}$: $y=1$ là có tương tác bất lợi đã được ghi nhận lâm sàng; $y=0$ là không có tương tác (hoặc chưa ghi nhận tương tác).

---

### [N] Tiêu chí 2: Mục tiêu và tiêu chí thành công
* **Câu hỏi GV:** Em chuyển mục tiêu thực tế thành mục tiêu học máy như thế nào? Mức kết quả mong muốn là bao nhiêu?
* **Trả lời mẫu:**
  - *Mục tiêu thực tế:* Cảnh báo sớm các phản ứng có hại khi bác sĩ kê đơn phối hợp nhiều thuốc (polypharmacy), đặc biệt là dự đoán tương tác cho các thuốc mới ra mắt lâm sàng (Cold-Start).
  - *Chuyển đổi thành mục tiêu học máy:* Xây dựng bộ phân loại nhị phân tối ưu hóa hàm mất mát trên các cặp thuốc, duy trì độ nhạy cảm cao và kiểm soát triệt để hiện tượng rò rỉ dữ liệu.
  - *Chỉ số mong muốn:*
    + Với kịch bản Warm-start: ROC-AUC $\ge 0{,}90$; F1-score $\ge 0{,}82$.
    + Với kịch bản khắt khe Strict Cold-start (cả hai thuốc mới): ROC-AUC $\ge 0{,}75$; PR-AUC (AUPR) $\ge 0{,}70$; Recall $\ge 0{,}75$ (để tránh bỏ sót tương tác nguy hiểm đe dọa tính mạng bệnh nhân).
    + So với Baseline ngẫu nhiên: ROC-AUC $\approx 0{,}50$.

---

### [N] Tiêu chí 3: Nguồn và mô tả dữ liệu
* **Câu hỏi GV:** Dữ liệu của em lấy từ đâu? Kích thước tập dữ liệu ra sao?
* **Trả lời mẫu:**
  - Dữ liệu kế thừa tập danh mục 548 thuốc chuẩn mực của nghiên cứu Rohani & Eslahchi (2019), cập nhật thông tin y sinh mới nhất từ CSDL DrugBank phiên bản 5.1.13, SIDER (tác dụng phụ) và OFFSIDES (biến cố bất lợi).
  - Không gian gồm đúng 548 thuốc, tạo thành $149.878$ cặp thuốc khả dĩ. Trong đó có $48.584$ cặp tương tác dương tính đã được chứng minh lâm sàng (mật độ $32{,}42\%$).
  - Có 10 kênh đặc trưng sinh học đa nguồn, trong đó cấu trúc phân tử được đo bằng chỉ số Tanimoto trên Morgan Fingerprints 2048-bit, văn bản chỉ định đo bằng TF-IDF + Cosine, và các tập hợp thuộc tính sinh học đo bằng chỉ số tương đồng Jaccard.

---

### [N] Tiêu chí 4: Khảo sát thống kê và phân bố dữ liệu
* **Câu hỏi GV:** Qua khảo sát thống kê, em thấy phân bố bậc tương tác và phân bố nhãn có đặc điểm gì nổi bật?
* **Trả lời mẫu:**
  - *Về phân bố nhãn:* Ban đầu dữ liệu mất cân bằng với tỷ lệ $1 : 2{,}08$ ($48.584$ mẫu dương vs $101.294$ mẫu âm). Sau khi lấy mẫu âm cân bằng 1:1, tập dữ liệu đạt quy mô $97.168$ mẫu cân bằng chuẩn $50\% - 50\%$.
  - *Về bậc tương tác thuốc:* Bậc trung bình là $177{,}31 \pm 129{,}64$ tương tác/thuốc, trung vị là $154{,}0$. Phân bố mạng mang tính chất không đồng nhất (scale-free): thuốc thấp nhất chỉ có 1 tương tác, trong khi thuốc trung tâm (Hub drug) có tới 513 tương tác (tương tác với $93{,}6\%$ tổng số thuốc còn lại).

---

### [N] Tiêu chí 5: Phân tích tương quan giữa các đặc trưng và với đầu ra
* **Câu hỏi GV:** Em hãy nêu 3 nhận xét có ý nghĩa cốt lõi rút ra từ ma trận tương quan và biểu đồ tương quan với nhãn?
* **Trả lời mẫu:**
  1. *Tính bổ trợ trực giao cao giữa các nguồn:* Tất cả hệ số tương quan giữa 10 kênh đều nằm dưới $0{,}29$ (mức cao nhất là giữa Target và Chem đạt $r=0{,}283$). Không có hai kênh nào bị đa cộng tuyến. Điều này chứng minh 10 nguồn phản ánh các góc nhìn sinh học độc lập, giải thích vì sao cần hợp nhất mạng (SNF) hoặc Ensemble đa góc nhìn.
  2. *Tương đồng sinh học tỷ lệ thuận với nguy cơ tương tác:* Ở các kênh chủ chốt (Target, Indication, Offside, Chem), các cặp có tương tác ($y=1$) có độ tương đồng cao hơn vượt trội so với cặp không tương tác ($y=0$). Cụ thể, độ tương đồng Target ở nhóm tương tác cao gấp $1{,}89$ lần nhóm âm ($0{,}0137$ vs $0{,}0072$).
  3. *Tác động của dữ liệu thưa:* Hai kênh Pathway và Transporter có giá trị nhóm âm cao hơn do ban đầu thiếu dữ liệu lớn (Pathway thiếu $63{,}69\%$), khẳng định sự cần thiết phải lọc đặc trưng Entropy và chuẩn hóa dữ liệu.

---

### [N] Tiêu chí 6 & 7: Định lượng và xử lý vấn đề chất lượng dữ liệu
* **Câu hỏi GV:** Dữ liệu có những vấn đề gì? Em đã xử lý ra sao và tại sao lại xử lý như vậy?
* **Trả lời mẫu:**
  - *Dữ liệu thiếu:* Pathway thiếu 349/548 thuốc ($63{,}69\%$), Transporter thiếu 217 ($39{,}60\%$), Enzyme thiếu 113 ($20{,}62\%$), Target thiếu 7, ATC thiếu 6, Chem thiếu 3.
  - *Cách xử lý missing:* Với các đặc trưng tập hợp y sinh, em gán tập rỗng $\emptyset$, Jaccard $= 0$. **Lý do khoa học:** Trong tin sinh học, tuyệt đối không được tự ý bịa đặt hoặc điền giá trị trung bình cho các phản ứng sinh hóa của thuốc khi y văn chưa ghi nhận. Thuật toán SNF sau đó sẽ tự động bù đắp thông tin từ các kênh có dữ liệu đầy đủ.
  - *Trùng lặp:* 0 bản ghi trùng; ma trận DDI đối xứng, đường chéo tự tương tác bằng 0.
  - *Mất cân bằng lớp:* Tỷ lệ ban đầu 1 : 2.08. Em xử lý bằng kỹ thuật **Balanced Negative Sampling 1:1** từ 101.294 cặp chưa quan sát, cố định seed=42 để thu được tập cân bằng 97.168 mẫu.
  - *Nguy cơ rò rỉ dữ liệu:* Chia ngẫu nhiên trên cạnh gây rò rỉ thực thể thuốc, em giải quyết bằng cách thiết kế 4 kịch bản chia dữ liệu phân tách thực thể thuốc.

---

### [N] Tiêu chí 8: Chuẩn bị dữ liệu và chống rò rỉ khi huấn luyện
* **Câu hỏi GV:** Em chia 4 kịch bản thực nghiệm như thế nào? Làm sao đảm bảo không bị rò rỉ dữ liệu (Data Leakage)?
* **Trả lời mẫu:**
  - *Case 1 (Warm Start):* Chia ngẫu nhiên theo cặp (Edge split) $\to$ Thuốc ở test có thể đã xuất hiện ở train $\to$ Mô hình bị rò rỉ node, kết quả thường bị lạc quan quá mức.
  - *Case 2 (Strict Cold Start - New $\times$ New):* Chia 548 thuốc thành $D_{\text{train}}$ (438 thuốc) và $D_{\text{test}}$ (110 thuốc) sao cho $D_{\text{train}} \cap D_{\text{test}} = \emptyset$. Test chỉ gồm các cặp tạo từ 110 thuốc mới ($3.456$ cặp cân bằng). Cả hai thuốc đều hoàn toàn mới đối với mô hình.
  - *Case 3 (Semi Cold Start - Old $\times$ New):* Test gồm các cặp chéo giữa 1 thuốc cũ và 1 thuốc mới ($30.154$ cặp cân bằng).
  - *Case 4 (Mix):* Gộp Case 2 và Case 3 ($33.610$ cặp cân bằng).
  - *Nguyên tắc Fit/Transform:* Toàn bộ các phép chuẩn hóa, trích xuất đặc trưng và chọn lọc đặc trưng bằng Entropy chỉ được tính toán trên tập Train (Fit), sau đó mới chuyển đổi (Transform) sang tập Test.

---

### [CN] Tiêu chí 9: Lý giải chiến lược đánh giá và hệ chỉ số
* **Câu hỏi GV:** Vì sao em không dùng phương pháp chia K-Fold ngẫu nhiên thông thường? Vì sao lại dùng chỉ số ROC-AUC và PR-AUC (AUPR) thay vì Accuracy?
* **Trả lời mẫu:**
  - *Vì sao không dùng chia ngẫu nhiên thông thường:* Chia ngẫu nhiên trên cặp thuốc chỉ đánh giá được bài toán transductive (thuốc đã biết). Trong thực tế, khi một loại thuốc mới được phê duyệt đưa vào bệnh viện, thuốc đó chưa từng có liên kết DDI nào. Chia ngẫu nhiên sẽ làm mô hình ghi nhớ biểu diễn nút (node memorization) thay vì học cơ chế hóa sinh. Do đó, phải chia theo thực thể thuốc rời rạc (Drug-disjoint node split) mới đo lường được khả năng tổng quát hóa thực sự.
  - *Vì sao không dùng Accuracy:* Không gian tương tác rất thưa. Nếu một mô hình dự đoán tất cả các cặp đều bằng 0, độ chính xác Accuracy vẫn đạt $> 67\%$, gây ngộ nhận nguy hiểm chết người.
  - *Vai trò của ROC-AUC:* Đo lường năng lực phân biệt trên toàn bộ miền ngưỡng xác suất, không bị phụ thuộc vào ngưỡng phân định $0{,}5$.
  - *Vai trò của PR-AUC (AUPR):* Đây là chỉ số quan trọng nhất trong y sinh vì nó tập trung trực tiếp vào lớp tương tác dương tính ($y=1$). Khi số lượng cặp âm rất lớn, PR-AUC đo lường khả năng tìm thấy tương tác thực sự mà không tạo ra quá nhiều cảnh báo giả làm phiền bác sĩ lâm sàng.

---

### [CN] Tiêu chí 10: Thao tác thực hành trực tiếp tại chỗ
* **Yêu cầu 1 của GV:** "Em hãy tra cứu thông tin và số lượng tương tác của một thuốc bất kỳ trên Notebook!"
  - **Thao tác:** Chạy ô `thong_ke_thuoc(0)` hoặc đổi chỉ số thuốc thành một số bất kỳ (ví dụ `thong_ke_thuoc(15)`):
  ```python
  thong_ke_thuoc(15)
  ```
  - **Giải thích:** Hàm sẽ lập tức in ra mã DrugBank ID, số lượng tương tác lâm sàng, cấu trúc SMILES, chỉ định điều trị và mã ATC của thuốc đó.

* **Yêu cầu 2 của GV:** "Em hãy đổi tỷ lệ lấy mẫu âm (Negative Sampling) từ 1:1 sang 1:2 và cho thầy/cô xem kết quả!"
  - **Thao tác:** Chạy hàm `doi_ty_le_negative_sampling(2.0)` tại mục 10.2 của Notebook:
  ```python
  doi_ty_le_negative_sampling(2.0)
  ```
  - **Giải thích:** Số lượng mẫu dương vẫn giữ nguyên $48.584$ mẫu ($33{,}3\%$), số lượng mẫu âm tăng lên $97.168$ mẫu ($66{,}7\%$), tổng số mẫu là $145.752$ mẫu.

* **Yêu cầu 3 của GV:** "Em hãy chứng minh tập Train và Test trong kịch bản Cold-Start của em không bị rò rỉ dữ liệu!"
  - **Thao tác:** Chạy hàm `chung_minh_khong_ro_ri(seed=42)` tại mục 10.3 của Notebook:
  ```python
  chung_minh_khong_ro_ri(SEED)
  ```
  - **Giải thích:** Output hiển thị rõ ràng: Tập Train gồm 438 thuốc ($80\%$), tập Test gồm 110 thuốc ($20\%$). Phép giao `Train ∩ Test = set()` (tập rỗng), chứng minh tuyệt đối không có sự trùng lặp thực thể thuốc giữa hai tập!

---

## III. TÓM TẮT CÁC CON SỐ "VÀNG" CẦN NHỚ KHI THUYẾT TRÌNH

| Khái niệm | Con số chính xác |
|---|---|
| Số lượng thuốc | **548** thuốc |
| Tổng số cặp thuốc khả dĩ | **149.878** cặp ($548 \times 547 / 2$) |
| Số cặp tương tác dương tính (DDI=1) | **48.584** cặp ($32{,}42\%$) |
| Số cặp chưa ghi nhận tương tác (DDI=0) | **101.294** cặp ($67{,}58\%$) |
| Bậc tương tác trung bình | **177,31** $\pm$ **129,64** tương tác |
| Thuốc có số tương tác nhiều nhất (Hub) | **513** tương tác ($93{,}6\%$ số thuốc) |
| Kênh thiếu dữ liệu nhiều nhất | **Pathway** (thiếu 349 thuốc -- $63{,}69\%$) |
| Kênh tương quan mạnh nhất | **Target** và **Chem** ($r = 0{,}283$) |
| Tập dữ liệu sau Negative Sampling 1:1 | **97.168** mẫu ($48.584$ dương + $48.584$ âm) |
| Kích thước thuốc phân chia | **438** thuốc Train ($80\%$) vs **110** thuốc Test ($20\%$) |
