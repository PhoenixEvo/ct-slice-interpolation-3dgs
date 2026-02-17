# Hướng dẫn tổng quát & chi tiết  
## Đề tài: Nội suy lát cắt CT 3D bằng 3D Gaussian Splatting (3DGS)

Mục tiêu: Xây dựng pipeline nội suy lát cắt (slice interpolation) trên volume CT public (ví dụ CT‑ORG – TCIA) bằng 3D Gaussian Splatting, có regularization phù hợp dữ liệu y khoa, và benchmark đúng chuẩn với các baseline hiện tại. [web:6][web:30][web:60]

---

## 1. Bối cảnh khoa học & ý nghĩa

- CT/MRI thường có độ phân giải cao trong mặt phẳng lát (in‑plane) nhưng độ dày lát (through‑plane) lớn → volume anisotropic, gây khó cho chẩn đoán 3D, planning, và segmentation. [web:30][web:32]  
- Slice interpolation (medical slice synthesis) nhằm “chèn” thêm lát giữa các lát đã có để tăng độ phân giải theo trục z mà không cần chụp thêm, giúp giảm thời gian chụp/ liều tia X. [web:30][web:55]  
- MedGS cho thấy 3D Gaussian Splatting có thể biểu diễn volume y khoa như chuỗi frame 2D trong không gian 3D, và dùng 3DGS để vừa nội suy frame vừa tái tạo surface/mesh. [web:6][web:9][web:17]  

**Ý nghĩa đề tài:**

- Mang 3DGS (vốn nổi trong 3D vision) vào bài toán y khoa thực tế. [web:6][web:45]  
- Xây pipeline đơn giản hơn MedGS nhưng vẫn bám đúng khung khoa học, có baseline mạnh (I³Net, interpolation cổ điển). [web:6][web:30]  

---

## 2. Dataset & chuẩn bị dữ liệu

### 2.1. Dataset chính: CT‑ORG (TCIA)

- CT‑ORG: 140 CT 3D, có nhãn 5 cơ quan (phổi, xương, gan, thận, bàng quang). [web:60][web:87]  
- Dữ liệu phong phú (các loại CT khác nhau) → tốt để đánh giá mô hình trong điều kiện thực tế. [web:60]  

**Cách dùng CT‑ORG cho slice interpolation:**

- Dùng volume CT (DICOM/NIfTI) làm ground truth full‑resolution theo trục z. [web:60]  
- Tự simulate sparse slices:
  - Ví dụ R=2: giữ slice số 0, 2, 4, …; slice số 1, 3, 5, … xem như ground truth cần nội suy. [web:30][web:55]  
  - R=3: giữ 0, 3, 6,…; nội suy 1, 2, 4, 5,… để tạo các mức khó khác nhau. [web:30][web:32]  

### 2.2. Dataset phụ (tùy thời gian)

Nếu còn thời gian, có thể thêm một bộ MRI não public từ các nguồn:

- Danh sách tổng hợp public MRI/CT: repo Medical‑Imaging‑Datasets. [web:41]  
- Survey các dataset MRI công khai cho nghiên cứu: tổng hợp nhiều bộ chế độ não và toàn thân. [web:38]  

Không bắt buộc cho version đầu, nhưng thêm MRI giúp đề tài “multi‑modal” hơn và giống tinh thần MedGS. [web:6][web:38]  

### 2.3. Tiền xử lý dữ liệu (gợi ý chuẩn khoa học)

- Chuẩn hóa intensity:
  - CT: clip HU (ví dụ từ −1000 đến 1000), sau đó normalize sang [0,1] hoặc mean‑std. [web:55]  
- Resample:
  - Có thể resample về độ dày lát cố định (vd 1 hoặc 2 mm) để đồng nhất giữa bệnh nhân (tùy tài nguyên). [web:55]  
- Split:
  - Train/val/test theo bệnh nhân, ví dụ 70/15/15, để tránh leakage giữa tập. [web:36][web:55]  

---

## 3. Kiến thức kỹ thuật cần nắm

### 3.1. 3D Gaussian Splatting cơ bản

- 3DGS biểu diễn scene/volume bằng tập Gaussian 3D: mỗi Gaussian có vị trí, ma trận hiệp phương sai (decompose thành scale + rotation) và thuộc tính (màu, opacity,…). [web:6][web:45]  
- Render: “splat” các Gaussian lên mặt phẳng nhìn (slice hoặc view) qua projection + alpha compositing. [web:6][web:20]  

Trong medical:

- MedGS biểu diễn các frame 2D (MRI/US) như các lát trong không gian 3D, dùng Gaussian để model chuyển tiếp giữa frame và surface. [web:6][web:9][web:17]  

### 3.2. Slice interpolation & I³Net

- I³Net giải quyết medical slice synthesis từ góc nhìn axial, tận dụng in‑plane cao để bù through‑plane thấp. [web:30][web:32]  
- Họ sử dụng cả inter‑slice (thông tin giữa các lát) và intra‑slice (texture trong 1 lát) để tối ưu. [web:30][web:83]  

Trong đề tài của bạn:

- Bạn không cần kiến trúc CNN phức tạp như I³Net, nhưng dùng I³Net làm baseline/tham khảo cấu hình thí nghiệm (cách simulate sparse, metric). [web:30][web:52]  

---

## 4. Thiết kế phương pháp 3DGS cho slice interpolation

### 4.1. Ý tưởng tổng quát

- Volume CT 3D → biểu diễn bằng tập Gaussian trong không gian (x, y, z). [web:6][web:20]  
- Đầu vào training:
  - Các slice “giữ lại” (ví dụ z=0, 2, 4,…) làm constraint.  
  - Mục tiêu: học tập Gaussians sao cho khi render ở các vị trí z trung gian (z=1, 3, 5,…) thì ảnh khớp với slice ground truth. [web:6][web:20]  

### 4.2. Khởi tạo Gaussian (khoa học nhưng đơn giản)

Một số chiến lược khả thi:

- Grid‑based:
  - Khởi tạo Gaussian tại mỗi voxel (hoặc patch), center = (x, y, z), intensity lấy từ CT. [web:6][web:20]  
  - Sau đó prune/merge dần để giảm số Gaussians (tùy tài nguyên).  
- Slice‑based:
  - Khởi tạo Gaussians trên các slice quan sát (z đã có ảnh), rồi cho phép chúng “lan” theo z trong training. [web:6][web:20]  

Bạn chọn phương án nào còn tùy GPU; grid‑based trên patch nhỏ thường dễ implement hơn với sinh viên.

### 4.3. Loss function và regularization

Thành phần loss chính:

- Reconstruction loss:
  - \(L_{\text{rec}} = \| S_{\text{pred}} - S_{\text{gt}} \|_1 \) hoặc L2 cho các slice cần nội suy. [web:6][web:20]  

Regularization theo hướng y khoa / không gian:

- Smoothness theo z:
  - Khuyến khích intensity/feature không thay đổi đột ngột giữa lát gần nhau (trừ khi có biên rõ). [web:6][web:20]  
- Edge/structure preservation:
  - Có thể thêm loss trên gradient (Sobel) để tránh làm mờ biên organ. [web:55]  

Tổng loss:

\[
L = L_{\text{rec}} + \lambda_{\text{smooth}} L_{\text{smooth}} + \lambda_{\text{edge}} L_{\text{edge}}
\]

(với \(\lambda\) chọn nhỏ, tune trên validation).  

---

## 5. Baseline & benchmark (thiết kế thực nghiệm chuẩn)

### 5.1. Baseline

- Classical interpolation:
  - Nearest, linear, cubic trên trục z (scipy / SimpleITK là đủ). [web:48][web:55]  
- Deep baseline:
  - U‑Net 2D (input: 2 lát trước–sau, output: lát giữa). [web:55]  
  - Nếu đủ thời gian: một phiên bản rút gọn ý tưởng I³Net (chỉ axial view, bớt cross‑view block). [web:30][web:32]  

### 5.2. Setup thí nghiệm

- Sparse ratio:
  - R=2, 3, có thể thêm R=4 giống các paper slice synthesis. [web:30][web:31]  
- Train/val/test:
  - Train và val trên subset bệnh nhân CT‑ORG, test trên nhóm bệnh nhân không trùng. [web:60][web:36]  

### 5.3. Metrics

- PSNR, SSIM trên các slice nội suy (toàn ảnh). [web:29][web:31][web:55]  
- Nếu dùng được mask CT‑ORG:
  - PSNR/SSIM trên ROI (gan, thận…) để xem chất lượng ở vùng quan trọng. [web:60][web:87]  
- Định tính:
  - Plot vài lát: GT vs linear vs U‑Net vs 3DGS, zoom vào biên organ. [web:55]  

---

## 6. Kế hoạch thực hiện (gợi ý 16–20 tuần)

### Giai đoạn 1 – Khởi động & đọc tài liệu (2–3 tuần)

Mục tiêu: hiểu bài toán + công nghệ.

- Đọc:
  - MedGS: cách họ dùng 3DGS cho multi‑modal 3D medical, tập trung phần interpolation. [web:6][web:9][web:17]  
  - I³Net: cách define task slice synthesis, cách simulate sparse & metric. [web:30][web:32][web:83]  
  - 1–2 bài về CT slice interpolation truyền thống (CT slice interpolation in the thorax, v.v.). [web:55][web:48]  
- Thiết lập:
  - Cài môi trường Python + PyTorch.  
  - Tải CT‑ORG từ TCIA, viết script đọc DICOM/NIfTI, normalize, resample (nếu cần). [web:60][web:87]  

Deliverable:  
- Ghi chú ngắn (1–2 trang) tóm tắt MedGS, I³Net, CT slice interpolation.  
- Script load & visualize vài volume CT‑ORG.

---

### Giai đoạn 2 – Baseline slice interpolation (3–4 tuần)

Mục tiêu: có baseline chạy được, làm nền so sánh.

- Implement:
  - Linear / cubic / nearest interpolation trên trục z (function rõ ràng). [web:48][web:55]  
  - U‑Net 2D nhỏ cho task (slice k, k+1 → slice k+0.5). [web:55]  
- Thiết lập pipeline thí nghiệm:
  - Viết code simulate sparse (R=2,3) từ volume. [web:30][web:32]  
  - Metric: PSNR/SSIM, implement và test trên vài volume. [web:29][web:31]  

Deliverable:  
- Bảng kết quả baseline (PSNR/SSIM) trên 10–20 volume.  
- 3–4 hình minh họa GT vs linear vs U‑Net.

---

### Giai đoạn 3 – 3DGS cho slice interpolation (4–5 tuần)

Mục tiêu: có phiên bản 3DGS đơn giản chạy được trên patch/volume nhỏ.

- Thiết kế biểu diễn:
  - Chọn kiểu khởi tạo Gaussian (grid‑based theo voxel, hoặc theo slice). [web:6][web:20]  
  - Viết hàm render slice từ 3D Gaussians (có thể reuse/đơn giản hóa từ code 3DGS open‑source). [web:6][web:18]  
- Implement loss:
  - L1/L2 reconstruction. [web:6][web:20]  
  - Smoothness theo z + optional edge‑loss. [web:6][web:55]  
- Training:
  - Bắt đầu train trên patch nhỏ (crop 128×128×D) để debug gradient và thời gian.  
  - Khi ổn, mở rộng sang toàn lát (nếu GPU cho phép).  

Deliverable:  
- 3DGS model chạy được, cho ra slice nội suy.  
- So sánh sơ bộ với linear/U‑Net trên 5–10 volume.

---

### Giai đoạn 4 – Benchmark đầy đủ & ablation (3–4 tuần)

Mục tiêu: hoàn thiện thực nghiệm khoa học.

- Benchmark:
  - Chạy 3DGS, linear, cubic, U‑Net trên full test set, với R=2 và R=3. [web:29][web:31][web:55]  
  - Ghi PSNR/SSIM tổng, + nếu có ROI thì thêm metric trên gan/thận. [web:60][web:87]  
- Ablation:
  - 3DGS không regularization vs có regularization. [web:6][web:20]  
  - (Tùy thời gian) số lượng Gaussian khác nhau.  
- Phân tích:
  - Vẽ histogram sai số, ví dụ lỗi theo z‑position.  
  - Case tốt/xấu, giải thích dựa trên cấu trúc giải phẫu.  

Deliverable:  
- Bảng kết quả chi tiết + hình định tính.  
- Ghi chú phân tích (1–2 trang).

---

### Giai đoạn 5 – Viết báo cáo / bài báo & chuẩn bị bảo vệ (2–3 tuần)

Mục tiêu: hoàn thiện sản phẩm khoa học.

Đề cương bài viết:

1. **Giới thiệu**  
   - Bài toán anisotropic CT/MRI, nhu cầu slice synthesis. [web:30][web:55]  
   - Hạn chế của interpolation cổ điển và CNN thuần 2D. [web:55][web:29]  
   - Đóng góp chính (3 bullet).  

2. **Related Work**  
   - Interpolation cổ điển (linear, sinc, optical flow). [web:48][web:55]  
   - Deep slice synthesis: I³Net, PixelMiner, các Net khác. [web:29][web:31][web:30]  
   - 3DGS & MedGS. [web:6][web:9][web:17]  

3. **Phương pháp**  
   - Mô hình 3DGS bạn dùng, khởi tạo, render slice. [web:6][web:20]  
   - Loss function, regularization.  
   - Tối ưu & cấu hình training.  

4. **Thực nghiệm**  
   - Dataset CT‑ORG, tiền xử lý. [web:60][web:87]  
   - Baseline & metric. [web:29][web:31][web:55]  
   - Kết quả định lượng & định tính, ablation.  

5. **Thảo luận & Hạn chế**  
   - Lợi thế & hạn chế của 3DGS trên CT.  
   - Hướng mở: multi‑modal (thêm MRI), faster GS, integration với segmentation. [web:6][web:38]  

6. **Kết luận**  

---

## 7. Gợi ý thực hành để làm hiệu quả

- Chia nhỏ milestone: luôn đảm bảo có baseline chạy được **trước** khi chơi 3DGS phức tạp. [web:55][web:30]  
- Giữ mọi thứ reproducible:
  - Log config (sparse ratio, learning rate, số Gaussian, seed).  
  - Dùng notebook/markdown để ghi lại thí nghiệm.  
- Bàn với thầy:
  - Sớm chốt phạm vi (chỉ CT‑ORG hay thêm MRI). [web:60][web:38]  
  - Thảo luận về việc có cần ethics/IRB không (vì đây là public de‑identified dataset, thường không cần, nhưng nên hỏi). [web:84]  

---

## 8. Checklist ngắn (tóm tắt)

1. Đọc MedGS + I³Net + 1 bài CT slice interpolation. [web:6][web:30][web:55]  
2. Tải & tiền xử lý CT‑ORG, viết script simulate sparse. [web:60][web:55]  
3. Implement baseline (linear, cubic, U‑Net) + metric. [web:48][web:29]  
4. Xây 3DGS rút gọn cho volume CT, với loss + regularization. [web:6][web:20]  
5. Chạy benchmark, ablation, lưu kết quả cẩn thận. [web:29][web:31][web:55]  
6. Viết report/bài báo theo outline.  

---
