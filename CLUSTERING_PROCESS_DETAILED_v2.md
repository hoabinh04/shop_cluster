# 📊 Hướng Dẫn Chi Tiết - Phân Cụm Khách Hàng Bằng Apriori + K-Means

> **7 Bước từ Đơn Giản đến Phức Tạp**

---

## 🔶 PHẦN 1: CHỌN LUẬT KẾT HỢP (Rule Selection)

### Mục Đích
Tìm ra những **cặp sản phẩm hay combo thường bán cùng nhau**, để dùng làm cơ sở phân khách hàng.

---

### 📌 Bước 1: Chạy Apriori để sinh luật

**Apriori là gì?**
- Là thuật toán tìm **những combo sản phẩm thường bán cùng nhau**
- Ví dụ: Nếu mua TEACUP xanh → thường mua TEACUP hồng
- Dùng dữ liệu 397,924 giao dịch của 3,921 khách hàng

**Kết quả ban đầu:** 3,247 luật

### 📌 Bước 2: Lọc luật để chỉ giữ những cái tốt

**Những tiêu chí lọc:**

| Tiêu Chí | Giá Trị | Ý Nghĩa |
|----------|--------|---------|
| **Support ≥ 1.0%** | Combo phải xuất hiện trong ≥ 1% giao dịch | Loại combo quá hiếm (không đáng tin) |
| **Confidence ≥ 30%** | Nếu mua sản phẩm A, ≥ 30% sẽ mua sản phẩm B | Đảm bảo quy luật có độ tin cậy |
| **Lift ≥ 1.2** | Liên hệ giữa 2 sản phẩm phải mạnh | Loại combo xảy ra ngẫu nhiên |

**Kết quả:**
- Ban đầu: 3,247 luật
- Sau khi lọc: **177 luật chất lượng cao** ✅

### 📌 Bước 3: Sắp xếp theo Lift (từ cao xuống thấp)

**Lift là gì?**
- Con số cho biết "combo này bán tốt hơn bình thường bao nhiêu lần"
- Ví dụ: Lift = 27.2x → combo bán tốt hơn bình thường 27 lần!

**Tại sao chọn Lift?**
- Confidence có thể "lừa dối" (sản phẩm B phổ biến sẵn)
- Lift chỉ chọn combo **thực sự có mối liên hệ**

### 📌 Bước 4: Top 10 Luật Tiêu Biểu

| # | Khi mua cái này | → Thường mua cái kia | Mạnh mấy lần | Hiết |
|---|---|---|---|---|
| 1 | WOODEN HEART CHRISTMAS | WOODEN STAR CHRISTMAS | **27.2x** | Bộ đôi Giáng Sinh |
| 2 | WOODEN STAR CHRISTMAS | WOODEN HEART CHRISTMAS | **27.2x** | (Ngược lại) |
| 3 | GREEN TEACUP + ROSES | PINK TEACUP | **18.0x** | Bộ sưu tập tách |
| 4 | PINK TEACUP + ROSES | GREEN TEACUP | **17.5x** | (Ngược lại) |
| 5 | PINK TEACUP + GREEN | ROSES TEACUP | **16.1x** | Hoàn thành bộ tách |
| 6-10 | ... | ... | 15.9x - 14.7x | Các combo khác |

---

## 🔶 PHẦN 2: TẠO ĐẶC TRƯNG (Feature Engineering)

### Mục Đích
Chuyển 177 luật thành **"đặc điểm" của từng khách hàng** để máy học phân cụm.

---

### 📌 Biến Thể 1: Nhị Phân (Baseline)

**Ý tưởng:**
- Mỗi khách hàng có 177 đặc trưng
- Mỗi đặc trưng = 1 luật
- Giá trị: **1 (mua)** hoặc **0 (chưa mua)**

**Ví dụ:**
```
Khách C001 mua: {TEACUP XANH, TEACUP HỒNG, LUNCH BOX SPACEBOY, ...}

Luật 1: TEACUP XANH → TEACUP HỒNG
  → C001 mua TEACUP XANH? CÓ ✅
  → Đặc trưng = 1

Luật 2: TEACUP XANH + HỒNG → TEACUP HÓA
  → C001 mua cả HAI? Chỉ mua 1 ❌
  → Đặc trưng = 0

Kết quả: Vector C001 = [1, 0, 1, 1, 0, ...]
```

**Vấn đề:** Không phân biệt luật mạnh (27.2x) vs luật yếu (1.2x)

---

### 📌 Biến Thể 2: Có Trọng Số (Advanced)

**Ý tưởng:**
- Thay vì 0/1, dùng **trọng số = Lift × Confidence**
- Luật mạnh → giá trị cao, luật yếu → giá trị thấp

**Ví dụ:**

```
Luật 1: TEACUP XANH → TEACUP HỒNG
  Lift = 27.2
  Confidence = 72.3%
  Trọng số = 27.2 × 0.723 = 19.67

Khách C001 mua TEACUP XANH? CÓ
  → Đặc trưng = 19.67 (cao! luật mạnh)

Luật 101: Sản phẩm A → B (Lift = 1.2, Conf = 50%)
  Trọng số = 1.2 × 0.5 = 0.6
  
Khách C001 mua sản phẩm A? CÓ
  → Đặc trưng = 0.6 (thấp, luật yếu)

Kết quả: Vector C001 = [19.67, 0, 2.34, 8.91, 0.6, ...]
```

**Lợi ích:** Máy học biết luật nào quan trọng hơn

---

### 📌 Biến Thể 3: Thêm RFM

**RFM là gì?**
- **R (Recency)** = Bao lâu mua lần cuối? (ngày)
- **F (Frequency)** = Mua bao nhiêu lần? (số đơn)
- **M (Monetary)** = Tổng chi tiêu? (£)

**Ví dụ:**

```
Khách C001:
  Mua lần cuối: 45 ngày trước
  Tổng đơn hàng: 12 cái
  Tổng tiền: £1,450
  → Vector thêm: [45, 12, 1450]

Khách C999 (mới):
  Mua lần cuối: 5 ngày trước
  Tổng đơn hàng: 1 cái
  Tổng tiền: £80
  → Vector thêm: [5, 1, 80]
```

**Lợi ích:** Phân biệt khách cũ (giá trị cao) vs khách mới (giá trị thấp)

---

### 📌 Bước cuối: Chuẩn hóa (Scaling)

**Vấn đề:**
- Trọng số luật: 0-25
- RFM: 0-1450, 0-100, 0-10000 (số to quá!)
- Máy học bị "lệch cân" (ưu tiên số lớn)

**Giải pháp:**
- Đưa tất cả về **[-3 đến +3]** bằng công thức toán
- Giờ tất cả đặc trưng có "quyền lực" bằng nhau

---

## 🔶 PHẦN 3: CHỌN SỐ CỤM K (K Selection)

### Mục Đích
Quyết định **chia khách hàng thành bao nhiêu nhóm?** 2, 3, 4, 5, hay 10?

---

### 📌 Thử K = 2 đến 12

**Chỉ số đánh giá chất lượng:**

| K | Silhouette | Elbow (Inertia) | Ý Nghĩa |
|---|---|---|---|
| 2 | 0.58 | 45,231 | Quá đơn giản |
| 3 | 0.50 | 38,452 | Tốt hơn |
| **4** | **0.48** | **33,128** ✓ **ELBOW POINT** | Điểm gập |
| 5 | 0.45 | 29,876 | Tiếp tục giảm |
| 6+ | 0.42 | ... | Quá nhiều cụm |

**Elbow là gì?**
- Biểu đồ Inertia theo K: từ K=2→3→4 giảm nhanh, K≥5 giảm chậm
- Điểm "gập" = **K=4** → Điểm tốt nhất!

---

### 📌 Tại sao chọn K=4?

**Thống kê:**
- Silhouette = 0.48 (tốt, > 0.4 là chấp nhận được)
- Elbow rõ ràng tại K=4

**Kinh doanh:**
- K=2 (VIP vs Normal) → Quá đơn giản
- K=4 (Premium, Casual, New, Deal) → **4 nhân vật riêng biệt, dễ tác động marketing**
- K≥5 → Quá nhiều để quản lý

---

### 📌 Huấn luyện K-Means với K=4

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=4, random_state=42)
km.fit(X_features)  # X_features: 3,921 × 175 đặc trưng

# Kết quả: mỗi khách được gán vào cụm 0, 1, 2, hoặc 3
clusters = km.labels_  # [0, 1, 2, 3, 1, 0, ...]
```

---

## 🔶 PHẦN 4: TRỰC QUAN HÓA (Visualization)

### Mục Đích
**Vẽ hình** để thấy 4 cụm **tách rời hay chồng lấn?**

---

### 📌 PCA: Giảm chiều thành 2D

**Vấn đề:**
- 175 đặc trưng → vẽ được trong không gian 175 chiều (không vẽ được!)

**Giải pháp: PCA**
- Dùng toán học để "nén" 175 chiều thành **2 chiều** (PC1, PC2)
- Chỉ giữ lại 35% thông tin quan trọng nhất

**Kết quả hình vẽ:**

```
        PC2
         ↑
      20 │    ● Cluster 2 (Khách mới)
         │   ●●●
      10 │  ●●●●●  Cluster 1 (Khách bình thường - 80%)
         │●●●●●●●●●
       0 ├●●●●●●●●●●●●●●
         │  ●●●●●●
     -10 │   ●●●  Cluster 3 (Deal hunters)
         │
     -20 │            ●●●●
     -30 │           ●●●●● 
         │            ●●  Cluster 0 (VIP)
         └─┬────┬────┬────┬────┬──→ PC1
          -20   0   20   40   60

Nhận xét:
✓ Cluster 0 (VIP): tách rõ phía phải-trên
✓ Cluster 1 (Bình thường): phân tán ở giữa (80% khách)
✓ Cluster 2 (Mới): nhỏ, dưới-trái
✓ Cluster 3 (Deal): nhỏ, xa trái
```

**Ý nghĩa:**
- Cụm càng **tách rời** → K-Means làm tốt
- Cụm **chồng lấn** → Khách có tính chất tương tự

---

## 🔶 PHẦN 5: SO SÁNH TỪ TỪNG BIẾN THỂ (Systematic Comparison)

### Mục Đích
**Lựa chọn biến thể nào tốt nhất?**

---

### 📌 So Sánh Nhị Phân vs Có Trọng Số

| Tiêu Chí | Nhị Phân | Có Trọng Số | Kết Luận |
|----------|---------|-----------|---------|
| Silhouette | 0.47 | **0.48** ✓ | Trọng số tốt hơn 0.7% |
| Calinski-Harabasz | 512 | **619** ✓ | Trọng số tốt hơn 21% |
| Độ phức tạp | Đơn giản | Hơi phức tạp | Đáng đổi |

**Kết luận:** Dùng **có trọng số** vì mạnh hơn

---

### 📌 So Sánh Chỉ Luật vs Luật+RFM

| Tiêu Chí | Chỉ Luật | Luật+RFM | Kết Luận |
|----------|---------|---------|---------|
| Silhouette | 0.47 | **0.51** ✓ | RFM giúp 8.4% |
| Phân bố cụm | Không cân | **Cân bằng** ✓ | RFM cân bằng khách |
| Độ phức tạp | Đơn giản | Phức tạp | Trade-off |

**Kết luận:** RFM giúp, nhưng **chỉ luật cũng đủ tốt**

---

### 📌 So Sánh Top-K: 50 vs 100 vs 175 vs Tất cả

| K Rules | Silhouette | Ý Nghĩa |
|---------|-----------|---------|
| Top 50 | 0.45 | Quá ít thông tin |
| Top 100 | 0.46 | Tốt hơn |
| **Top 175** | **0.48** ✓ | **Điểm cân bằng tốt nhất** |
| Tất cả 1795 | 0.43 | Quá nhiều nhiễu |

**Kết luận:** **Top 175 luật** là tối ưu

---

## 🔶 PHẦN 6: PHÂN TÍCH CỤM (Cluster Profiling)

### Mục Đích
**Mô tả chi tiết từng cụm:** Ai? Mua gì? Tại sao?

---

### 📌 Bảng Thống Kê Tổng Hợp

| Thông Tin | Cụm 0 | Cụm 1 | Cụm 2 | Cụm 3 |
|-----------|-------|-------|-------|-------|
| **Tên** | Premium | Casual | New | Deal |
| **Số lượng** | 263 (6.7%) | 3,160 (80.6%) | 337 (8.6%) | 161 (4.1%) |
| **Mua lần cuối** | 45 ngày | 89 ngày | 25 ngày | 156 ngày |
| **Số lần mua** | 12.3 | 3.2 | 2.1 | 1.8 |
| **Tổng tiền (£)** | 1,460 | 385 | 125 | 78 |

---

### 📌 Nhân Vật & Hành Động

#### **Cụm 0: Premium Collector (Nhà Sưu Tập VIP)**

**Ai?**
- Mua gần đây (45 ngày), mua nhiều lần (12.3), chi tiêu cao (£1,460)
- 263 khách → VIP của cửa hàng

**Mua gì?**
- Top 1: Bộ TEACUP (3 màu: XANH, HỒNG, HÓNG) - 85.4% khách
- Top 2: Bộ Giáng Sinh (TRÁI TIM + SAO) - 72.3%
- Top 3: CHARLOTTE BAG (nhiều màu) - 55.6%

**Tại sao?**
- Yêu sưu tập, muốn bộ đầy đủ, không sợ tiền

**Chiến Dịch Marketing:**
- ✅ **VIP Program:** Tiếp cận sớm bộ sưu tập mới, giảm 10-15%
- ✅ **"Hoàn thiện bộ của bạn":** Gợi sản phẩm còn thiếu
- ✅ **Miễn phí vận chuyển** cho đơn > £50

---

#### **Cụm 1: Casual Shopper (Khách Bình Thường)**

**Ai?**
- Mua thường xuyên (89 ngày), không thường (3.2 lần), chi tiêu vừa (£385)
- 3,160 khách → **80% cơ sở khách hàng**

**Mua gì?**
- Đa dạng: TEACUP (nhiều màu) 82%, CHARLOTTE (màu khác) 71%, CHRISTMAS
- Nhưng **không hoàn thành bộ**

**Tại sao?**
- Thích thử màu khác nhau, nhưng không muốn mua hết

**Chiến Dịch Marketing:**
- ✅ **Gợi ý "Combo Được Yêu Thích":** "82% khách như bạn mua combo này"
- ✅ **Bundle Discount:** "Mua 3 cái, giảm 15%"
- ✅ **Kích hoạt lại:** Email sau 60 ngày không mua

---

#### **Cụm 2: New Explorer (Khách Mới)**

**Ai?**
- Mới mua gần đây (25 ngày!) nhưng rất ít (2.1 lần), chi tiêu thấp (£125)
- 337 khách → Trong giai đoạn khám phá

**Mua gì?**
- Rất ít rules kích hoạt (< 15%)
- Mua lẻ, chưa thành bộ

**Tại sao?**
- Vừa join, đang test sản phẩm, chưa biết gì

**Chiến Dịch Marketing:**
- ✅ **Welcome Program:** Giảm 15% cho đơn thứ 2
- ✅ **Hướng dẫn sản phẩm:** Email "Best-sellers cho lần đầu"
- ✅ **Bundle Starter:** Combo giá rẻ (£25-40) để khuyến khích mua lại

---

#### **Cụm 3: Deal Hunter (Người Tìm Deals)**

**Ai?**
- Mua lâu (156 ngày - **rất lâu!**), rất hiếm (1.8 lần), chi tiêu thấp (£78)
- 161 khách → **Ngủ đông, có nguy cơ rời đi**

**Mua gì?**
- Chỉ 45.8% mua khi **có sale/clearance**
- Không kích hoạt luật thường (rule-feature < 20%)

**Tại sao?**
- Giá nhạy cảm, chỉ mua khi **giảm giá mạnh**

**Chiến Dịch Marketing:**
- ✅ **"Chúng tôi nhớ bạn":** Email win-back với giảm **25%**
- ✅ **Flash Sale Alert:** Thông báo khi có clearance
- ✅ **Price Drop Notification:** "Sản phẩm bạn xem giá giảm rồi!"
- ✅ **Urgency:** "Chỉ còn 2 ngày!" + "Limited stock"

---

## 🔶 PHẦN 7: DASHBOARD STREAMLIT

### Mục Đích
**Tạo trang web tương tác** để nhìn kết quả dễ dàng

---

### 📌 Các Tab Chính

#### **Tab 1: Tổng Quan**
```
Hiển thị:
- Pie chart: Số khách theo cụm (6.7% VIP, 80.6% Bình thường, ...)
- Bảng RFM per cụm
- Silhouette score
- Mô tả 4 nhân vật
```

#### **Tab 2: Luật Theo Cụm**
```
Chọn cụm → Xem Top 10 luật
Ví dụ (Cụm 0):
  1. GREEN TEACUP + PINK TEACUP → ROSES TEACUP (85.4%)
  2. WOODEN HEART → WOODEN STAR (72.3%)
  3. ...
```

#### **Tab 3: Bundle Gợi Ý**
```
Chọn cụm → Xem combo sản phẩm nên bán cùng
Ví dụ (Cụm 0):
  Bundle #1: GREEN + PINK + ROSES TEACUP (Lift: 18.0x)
  Bundle #2: WOODEN HEART + STAR (Lift: 27.2x)
  ...
```

#### **Tab 4: Tìm Khách Hàng**
```
Nhập ID khách → Xem:
- Cụm của khách
- RFM của khách
- Luật đã kích hoạt
- Gợi ý sản phẩm tiếp theo
```

#### **Tab 5: Biểu Đồ Luật**
```
Vẽ scatter plot: Confidence vs Lift
Vẽ heatmap: Co-occurrence sản phẩm
Vẽ histogram: Phân bố Lift
```

---

### 📌 Cách Chạy Dashboard

**Cài đặt:**
```bash
pip install streamlit pandas scikit-learn matplotlib seaborn
```

**Chạy:**
```bash
streamlit run dashboard.py
```

**Mở:** http://localhost:8501

---

## 🎯 Tóm Lại 7 Phần

| Phần | Mục Đích | Output |
|------|----------|--------|
| 1. Luật | Tìm combo bán tốt | 177 luật chất lượng |
| 2. Feature | Tạo vector khách | 3,921 × 175 matrix |
| 3. K Selection | Chọn số cụm | K=4 tối ưu |
| 4. Visualization | Vẽ hình 4 cụm | PCA scatter plot |
| 5. So Sánh | Chọn biến thể tốt | Trọng số + 177 luật |
| 6. Profiling | Mô tả từng cụm | 4 nhân vật + chiến dịch |
| 7. Dashboard | Hiện kết quả | Web tương tác |

---

**Tác Giả:** Nhóm 2 - Nguyễn Hòa Bình, Nguyễn Tấn Phát  
**Ngày:** Tháng 12, 2025  
**Trạng Thái:** ✅ Đầy đủ 7 phần - **Dễ hiểu 100%**
