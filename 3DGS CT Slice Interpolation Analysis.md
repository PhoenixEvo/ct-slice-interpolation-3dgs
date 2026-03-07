# Phân tích đề tài: CT Slice Interpolation via 3D Gaussian Splatting

## 1. Tổng quan dự án

Dựa trên README đã đọc, dự án xây dựng một pipeline nội suy lát cắt CT 3D bằng 3D Gaussian Splatting (3DGS) trên bộ dữ liệu CT-ORG (TCIA). Kiến trúc chính sử dụng **axis-aligned 3D Gaussians** (không rotation, chỉ position + log-scale + opacity + intensity), với **differentiable slice renderer** dạng tile-based, **medical-specific regularization** (smoothness theo z + edge preservation), và **adaptive densification/pruning**. Dự án so sánh với baseline classical (nearest/linear/cubic) và U-Net 2D, đánh giá bằng PSNR, SSIM, ROI-based metrics, và ablation study.

***

## 2. Bối cảnh nghiên cứu & cơ sở khoa học

### 2.1. Bài toán CT slice interpolation

CT/MRI volume thường có **anisotropy** rõ rệt: in-plane resolution cao (0.5–1 mm) nhưng through-plane (trục z) thấp hơn nhiều (1–5 mm hoặc hơn). Điều này gây khó khăn cho:[^1][^2]

- Phân đoạn 3D (segmentation) và tái tạo bề mặt cơ quan
- Chẩn đoán dựa trên ảnh coronal/sagittal (bị nhòe)
- Planning phẫu thuật và xạ trị

**Slice interpolation** (hay slice synthesis) nhằm tạo thêm lát ở giữa các lát đã chụp mà không tăng liều bức xạ. Đây là bài toán đã có lịch sử nghiên cứu dài nhưng vẫn còn nhiều thách thức, đặc biệt ở vùng biên cơ quan và vùng có biến đổi giải phẫu nhanh theo z.[^2][^1]

### 2.2. Các phương pháp nội suy truyền thống

- **Linear/cubic interpolation**: Nhanh nhưng gây mờ (blurring) và artifact ở biên cơ quan, đặc biệt khi khoảng cách lát lớn.[^1][^2]
- **Optical flow-based**: Tốt hơn nhưng phức tạp, không phải lúc nào cũng ổn định trên CT.[^2]

### 2.3. Deep learning cho slice interpolation

- **Parallel U-Net** (Wu et al., PLOS ONE 2022): Dùng nhiều U-Net song song để tái tạo nhiều lát giữa từ 2 lát liền kề. Giảm MAE 22.05% so với linear interpolation, cải thiện thêm 15% cho gan với kỹ thuật range-clip. Thử nghiệm trên 130 bệnh nhân.[^1][^2]
- **I³Net** (2024): Inter-Intra-slice Interpolation Network, khai thác cả thông tin trong lát (intra-slice) và giữa lát (inter-slice) cho medical slice synthesis. Là baseline mạnh gần đây.[^3][^4]
- **Through-plane super-resolution** (MRI): Nhiều nghiên cứu dùng CNN/Transformer để tăng resolution theo z cho MRI, tuy nhiên chủ yếu dành cho MRI chứ không tập trung vào CT.[^5]

***

## 3. Các công trình liên quan về 3DGS trong y khoa

### 3.1. MedGS (Sep 2025)

MedGS là **framework GS-based đầu tiên cho multi-modal 3D medical imaging**. Các đặc điểm chính:

- Biểu diễn dữ liệu y khoa (US, MRI) dưới dạng chuỗi frame 2D trong không gian 3D, model bằng Gaussian-based distributions.[^6][^7]
- Dùng Folded-Gaussian primitives và VeGaS formulation để nội suy frame và tái tạo surface/mesh.[^6]
- Kết quả cho thấy GS vượt trội cả interpolation cổ điển và neural approaches trên US và MRI.[^6]
- **Hạn chế**: Chưa áp dụng cho CT slice interpolation cụ thể; kiến trúc khá phức tạp (Folded-Gaussian + VeGaS) → không tối ưu cho CT anisotropic đơn giản.

### 3.2. 3DGR-CT (Medical Image Analysis 2025)

- Dùng 3D Gaussian representation cho **sparse-view CT reconstruction** (ít góc chụp), khác hoàn toàn với slice interpolation (ít lát).[^8][^9][^10]
- Hai đổi mới chính: (i) FBP-image-guided Gaussian initialization, (ii) differentiable CT projector.[^9][^10]
- Kết quả vượt state-of-the-art INR methods (NeRF-based), training nhanh hơn.[^10][^8]
- **Khác biệt với dự án**: 3DGR-CT giải quyết bài toán angular projection (sparse-view reconstruction), không phải through-plane slice interpolation.

### 3.3. DDGS-CT (NeurIPS 2024)

- Direction-Disentangled Gaussian Splatting cho Digitally Reconstructed Radiograph (DRR) generation từ CT.[^11][^12]
- Tách radiosity thành isotropic + direction-dependent components, phù hợp physics X-ray.[^12]
- **Khác biệt**: DDGS-CT tạo DRR (ảnh X-ray 2D), không phải nội suy lát cắt CT.

### 3.4. Axis-Aligned Gaussian Splatting (AAGS)

- Gong et al. đề xuất giới hạn Gaussian kernels chỉ theo 3 hướng trục (x, y, z), giảm tham số và tăng hiệu quả.[^13]
- Dự án của bạn sử dụng tư tưởng tương tự (axis-aligned, no rotation) → phù hợp cho CT vì volume CT tự nhiên aligned theo trục.

### 3.5. Các paper 3DGS medical khác (MICCAI 2024–2025)

Tại MICCAI 2024, nhiều paper áp dụng 3DGS cho y khoa nhưng chủ yếu tập trung vào **surgical scene reconstruction** (EndoGaussian, SurgicalGaussian, Free-SurGS) và **sparse-view cone-beam CT**. Không có paper nào cụ thể về CT slice interpolation bằng 3DGS.[^14][^15][^16]

***

## 4. Phân tích khe hở nghiên cứu (Research Gap)

Đây là phần quan trọng nhất để đánh giá đề tài có "viết paper được không":

| Công trình | Bài toán | Modality | 3DGS? | Slice interp.? |
|---|---|---|---|---|
| MedGS[^6] | Frame interpolation + surface recon | US, MRI | ✅ (Folded-GS) | ✅ (nhưng US/MRI, phức tạp) |
| 3DGR-CT[^10] | Sparse-view CT recon (angular) | CT | ✅ (3DGR) | ❌ |
| DDGS-CT[^12] | DRR generation | CT | ✅ (DDGS) | ❌ |
| I³Net[^3] | Slice interpolation | CT/MRI | ❌ (CNN) | ✅ |
| Parallel U-Net[^2] | Slice interpolation | CT | ❌ (CNN) | ✅ |
| **Dự án này** | **Slice interpolation** | **CT** | **✅ (axis-aligned)** | **✅** |

**Khe hở rõ ràng**: Chưa có công trình nào áp dụng 3D Gaussian Splatting (dạng axis-aligned, đơn giản hóa) cho bài toán **CT through-plane slice interpolation** cụ thể. MedGS gần nhất nhưng dùng cho US/MRI với kiến trúc phức tạp hơn. Các paper 3DGS khác trong CT (3DGR-CT, DDGS-CT) giải quyết bài toán khác (sparse-view angular reconstruction, DRR).

→ **Đề tài có novelty đủ để viết paper** ở mức workshop/conference regional hoặc journal nếu kết quả tốt.

***

## 5. Đóng góp học thuật (Contributions)

Dựa trên README và phân tích gap, đề tài có thể claim các đóng góp sau:

### Contribution 1: Pipeline 3DGS đầu tiên cho CT slice interpolation
Đề xuất pipeline per-volume 3DGS optimization với axis-aligned Gaussians cho bài toán nội suy lát cắt CT, khác biệt với MedGS (US/MRI, Folded-GS) và 3DGR-CT (sparse-view reconstruction).[^10][^6]

### Contribution 2: Medical-specific regularization
Thiết kế loss function kết hợp reconstruction loss với:
- **Inter-slice smoothness**: ràng buộc Gaussians liền kề theo z không thay đổi đột ngột
- **Edge preservation**: bảo tồn biên cơ quan (gradient-based)

Điều này khác với regularization trong 3DGS gốc (vốn thiết kế cho natural scenes).[^2][^6]

### Contribution 3: Benchmark hệ thống trên CT-ORG
So sánh có hệ thống giữa 3DGS, classical interpolation, và deep learning baseline (U-Net) trên bộ CT-ORG public với cả whole-image metrics và ROI-based metrics (per-organ evaluation).[^17][^18][^2]

***

## 6. Đánh giá tính khả thi

### 6.1. Điểm mạnh (Feasible)

- **Axis-aligned simplification**: Bỏ rotation giảm đáng kể số tham số và phức tạp gradient, phù hợp với bản chất CT volume (vốn aligned theo trục).[^13]
- **CT-ORG đã sẵn sàng**: Public, đa dạng (140 CT, nhiều điều kiện chụp), có organ mask để đánh giá ROI → không cần xin IRB, tải ngay được.[^19][^20][^17]
- **Baseline rõ ràng**: Linear/cubic có sẵn (scipy), U-Net có nhiều implementation → dễ reproduce và so sánh công bằng.[^2]
- **GPU L4 (24GB)**: Đủ cho per-volume optimization nếu dùng mixed precision và tile-based rendering.[^21]

### 6.2. Rủi ro và cách giảm thiểu

**Rủi ro 1: 3DGS có thể không thắng cubic interpolation rõ ràng ở R=2**

- R=2 (bỏ 1 lát, giữ 1 lát) thì khoảng cách ngắn → cubic đã khá tốt. Sự khác biệt có thể nhỏ.
- **Giải pháp**: Tập trung vào R=3 hoặc R=4 (sparse hơn) để 3DGS thể hiện ưu thế rõ hơn. Đồng thời, nhấn mạnh kết quả trên vùng biên cơ quan (ROI) nơi cubic thường bị mờ nhất.[^2]

**Rủi ro 2: Per-volume optimization chậm**

- Phải train riêng cho mỗi volume → tốn thời gian khi chạy trên 140 volumes.
- **Giải pháp**: Chạy trên subset (30–50 volumes) cho thí nghiệm chính, báo cáo runtime trung bình. Đây cũng là hạn chế cần nêu rõ trong paper.

**Rủi ro 3: So sánh không công bằng với U-Net**

- U-Net là generalizable (train 1 lần, test nhiều volumes) trong khi 3DGS là per-volume → U-Net nhanh hơn ở inference.
- **Giải pháp**: Báo cáo cả chất lượng VÀ runtime. Nếu 3DGS quality cao hơn, argue rằng per-volume optimization có lợi cho các trường hợp cần chất lượng cao (surgical planning, research). Nếu U-Net tốt hơn cả về quality, cần xem xét lại approach.

***

## 7. Kết quả cần đạt để viết paper

### 7.1. Kết quả định lượng bắt buộc

**Bảng chính (Table 1)**: So sánh PSNR/SSIM giữa các phương pháp

| Method | R=2 PSNR | R=2 SSIM | R=3 PSNR | R=3 SSIM |
|---|---|---|---|---|
| Nearest | ~25–28 | ~0.80–0.85 | ~22–25 | ~0.75–0.80 |
| Linear | ~30–33 | ~0.90–0.93 | ~27–30 | ~0.85–0.90 |
| Cubic | ~31–34 | ~0.91–0.94 | ~28–31 | ~0.86–0.91 |
| U-Net 2D | ~33–36 | ~0.93–0.96 | ~30–33 | ~0.90–0.93 |
| **3DGS (Ours)** | **~34–37** | **~0.94–0.97** | **~31–34** | **~0.91–0.94** |

*Lưu ý: Các giá trị trên là ước lượng dựa trên benchmark tương tự trong literature. Bạn cần chạy thực nghiệm để có số chính xác.*[^3][^2]

Để paper có sức thuyết phục, 3DGS cần đạt **ít nhất ngang hoặc tốt hơn U-Net** trên PSNR/SSIM, đặc biệt ở R=3.

**Bảng ROI (Table 2)**: Per-organ metrics (dùng mask CT-ORG)

| Method | Liver PSNR | Kidney PSNR | Lung PSNR |
|---|---|---|---|
| Cubic | ... | ... | ... |
| U-Net | ... | ... | ... |
| **3DGS (Ours)** | ... | ... | ... |

Đây là điểm mạnh của đề tài: CT-ORG có mask cơ quan nên bạn có thể đánh giá trên từng vùng quan trọng. Nếu 3DGS tốt hơn ở biên cơ quan (gan, thận) dù chỉ nhỉnh hơn 0.5–1 dB, đó đã là đóng góp có ý nghĩa y khoa.[^18][^19][^17]

### 7.2. Ablation study bắt buộc

| Variant | PSNR | SSIM |
|---|---|---|
| 3DGS (no reg) | ... | ... |
| 3DGS + smoothness only | ... | ... |
| 3DGS + edge only | ... | ... |
| **3DGS + both (full)** | ... | ... |

Ablation phải cho thấy:
- Regularization cải thiện so với không có (chứng minh contribution 2).
- Cả hai loại regularization đều đóng góp (không thừa).

### 7.3. Kết quả định tính bắt buộc

- **Figure so sánh**: Chọn 3–4 lát CT có vùng biên cơ quan rõ (gan, thận), hiển thị GT vs Cubic vs U-Net vs 3DGS, zoom vào biên. 3DGS cần cho thấy biên sắc nét hơn, ít blurring hơn.
- **Error map**: Hiển thị absolute error map cho từng phương pháp trên cùng 1 slice.

### 7.4. Kết quả phụ (nice-to-have)

- Runtime comparison (training time per volume, inference time per slice).
- Convergence curve (PSNR vs iteration).
- Nếu thêm 1 bộ MRI → chứng minh generalization across modality.

***

## 8. Hướng phát triển và mở rộng

### 8.1. Ngắn hạn (trong đề tài)
- Thử thêm R=4 hoặc R=5 để xem 3DGS có robust hơn baseline ở mức sparse cao không.
- Thử trên subset MRI (nếu có thời gian) để claim "multi-modal" nhẹ.

### 8.2. Trung hạn (nếu muốn nâng lên conference paper)
- **Generalizable 3DGS**: Thay vì per-volume optimization, train một encoder dự đoán Gaussian parameters cho volume mới (feed-forward), giảm inference time.
- **Kết hợp segmentation**: Dùng Gaussian representation để đồng thời nội suy slice VÀ segment cơ quan.

### 8.3. Dài hạn
- Áp dụng cho **4D CT** (CT theo thời gian, ví dụ hô hấp) với deformable Gaussians.
- Kết hợp với **diffusion models** để tăng chất lượng texture ở vùng khó.

***

## 9. Nơi có thể submit paper

Tùy chất lượng kết quả:

- **Kết quả tốt** (3DGS > U-Net rõ ràng, ablation mạnh): Workshop tại MICCAI, ISBI, hoặc conference như ACCV, RIVF (hội nghị phù hợp sinh viên Việt Nam).
- **Kết quả khá** (3DGS ≈ U-Net nhưng có insight thú vị): RIVF, KSE, hoặc journal Việt Nam (Tạp chí Khoa học & Công nghệ).
- **Kết quả xuất sắc**: MICCAI workshop, Medical Image Analysis (journal), hoặc IEEE TMI short paper.

***

## 10. Tóm tắt đánh giá

| Tiêu chí | Đánh giá |
|---|---|
| Khe hở nghiên cứu | ✅ Rõ ràng – chưa có paper 3DGS cho CT slice interpolation cụ thể |
| Novelty | ✅ Axis-aligned 3DGS + medical regularization cho CT anisotropy |
| Tính khả thi kỹ thuật | ✅ Cao – dataset public, baseline sẵn, GPU L4 đủ |
| Rủi ro | ⚠️ Trung bình – cần 3DGS thắng cubic/U-Net ít nhất ở R≥3 |
| Đóng góp viết paper | ✅ Đủ cho workshop/conference nếu kết quả rõ |
| Giá trị thực tiễn | ✅ Giảm anisotropy CT, hỗ trợ chẩn đoán/planning |

Đề tài hoàn toàn có thể viết thành paper nếu kết quả thực nghiệm cho thấy 3DGS cải thiện so với baseline, đặc biệt ở vùng biên cơ quan và ở mức sparse cao (R≥3). Điểm mấu chốt là **chạy thực nghiệm sớm** để biết hướng đi có khả thi không, rồi mới dồn sức viết.

---

## References

1. [Computed Tomography slice interpolation in the longitudinal direction based on deep learning techniques: To reduce slice thickness or slice increment without dose increase](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9754169/) - Large slice thickness or slice increment causes information insufficiency of Computed Tomography (CT...

2. [Computed Tomography slice interpolation in the longitudinal ...](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0279005) - Large slice thickness or slice increment causes information insufficiency of Computed Tomography (CT...

3. [Inter-Intra-slice Interpolation Network for Medical Slice Synthesis](https://arxiv.org/html/2405.02857v1)

4. [I³Net: Inter-Intra-Slice Interpolation Network for Medical Slice Synthesis](https://pubmed.ncbi.nlm.nih.gov/38669167/) - We propose an Inter-Intra-slice Interpolation Network ( [Formula: see text]Net), which fully explore...

5. [MRI Super-Resolution with Deep Learning](https://arxiv.org/html/2511.16854v1) - This approach is widely used in medical image analysis, where variations in resolution, contrast, or...

6. [MedGS: Gaussian Splatting for Multi-Modal 3D Medical Imaging](https://arxiv.org/html/2509.16806v1)

7. [Gaussian Splatting for Multi-Modal 3D Medical Imaging](https://arxiv.org/abs/2509.16806) - Multi-modal three-dimensional (3D) medical imaging data, derived from ultrasound, magnetic resonance...

8. [3DGR-CT: Sparse-view CT reconstruction with a 3D...](https://www.ovid.com/journals/meian/abstract/10.1016/j.media.2025.103585~3dgr-ct-sparse-view-ct-reconstruction-with-a-3d-gaussian) - ABSTRACTSparse-view computed tomography (CT) reduces radiation exposure by acquiring fewer projectio...

9. [3DGR-CT: Sparse-View CT Reconstruction with a 3D ...](https://arxiv.org/abs/2312.15676) - Sparse-view computed tomography (CT) reduces radiation exposure by acquiring fewer projections, maki...

10. [Sparse-View CT Reconstruction with a 3D Gaussian Representation](https://arxiv.org/html/2312.15676v2)

11. [DDGS-CT: Direction-Disentangled Gaussian Splatting for ...](https://neurips.cc/virtual/2024/poster/93752) - We present a novel approach that marries realistic physics-inspired X-ray simulation with efficient,...

12. [DDGS-CT: Direction-Disentangled Gaussian Splatting for ...](https://arxiv.org/abs/2406.02518) - by Z Gao · 2024 · Cited by 24 — We present a novel approach that marries realistic physics-inspired ...

13. [Axis-Aligned Gaussian Splatting for Radiance Fields](https://dl.acm.org/doi/10.1145/3675018.3675770) - The proposed axis-aligned Gaussian splatting method offers a middle ground by restricting kernels to...

14. [3D-Gaussian-Splatting-Papers/MICCAI2024.md at main · Awesome3DGS/3D-Gaussian-Splatting-Papers](https://github.com/Awesome3DGS/3D-Gaussian-Splatting-Papers/blob/main/MICCAI2024.md) - 3D高斯论文，持续更新，欢迎交流讨论。. Contribute to Awesome3DGS/3D-Gaussian-Splatting-Papers development by creating ...

15. [EndoGaussian: Gaussian Splatting for Deformable Surgical ...](https://yifliu3.github.io/EndoGaussian/) - EndoGaussian: Gaussian Splatting for Deformable Surgical Scene Reconstruction.

16. [SurgicalGaussian: Deformable 3D Gaussians for High-Fidelity ...](https://surgicalgaussian.github.io) - We develop SurgicalGaussian, a deformable 3D Gaussian Splatting method to model dynamic surgical sce...

17. [CT-ORG - The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/ct-org/) - This dataset consists of 140 computed tomography (CT) scans, each with five organs labeled in 3D: lu...

18. [CT-ORG, a new dataset for multiple organ segmentation in ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC7658204/) - Despite the relative ease of locating organs in the human body, automated organ segmentation has bee...

19. [Download (2.94kb)](https://www.cancerimagingarchive.net/wp-content/uploads/ctorg_README.txt) - This dataset consists of 140 computed tomography (CT) scans, each with five organs labeled in 3D: lu...

20. [arekborucki/CADS-dataset at main](https://huggingface.co/datasets/arekborucki/CADS-dataset/blob/main/0008_ctorg/README_0008_ctorg.md) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

21. [Volume-based Gaussian Splatting](https://www.emergentmind.com/topics/volume-based-gaussian-splatting) - Volume-based Gaussian Splatting enables efficient, real-time volumetric rendering and scene reconstr...

