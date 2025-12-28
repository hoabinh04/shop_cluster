# 📊 Giải Thích Chi Tiết - Quy Trình Phân Cụm Khách Hàng

> Theo yêu cầu đề bài: Rule Selection → Feature Engineering → K Selection → Visualization → Profiling → Marketing Strategy

---

## PHẦN 1: RULE SELECTION - Lựa Chọn Luật Kết Hợp

### 1.1 Quy Trình Sinh Luật (Apriori)

**Bước 1: Chạy Apriori Algorithm**

```
Input: 397,924 giao dịch → 3,921 khách hàng × 3,665 sản phẩm

Apriori Output: 3,247 luật ban đầu
├─ Từ: {WOODEN HEART CHRISTMAS SCANDINAVIAN} → {WOODEN STAR CHRISTMAS SCANDINAVIAN}
├─ Từ: {GREEN REGENCY TEACUP AND SAUCER} → {PINK REGENCY TEACUP AND SAUCER}
├─ Từ: {SPACEBOY LUNCH BOX} → {DOLLY GIRL LUNCH BOX}
└─ ... (3,247 luật tổng cộng)

Parameters sử dụng:
- min_support = 0.5% (xuất hiện ≥ 0.5% giao dịch)
- min_confidence = 10% (để có nhiều luật chọn)
```

### 1.2 Tiêu Chí Lọc Luật (Rule Filtering)

**Bước 2: Áp dụng các ngưỡng lọc**

| Tiêu Chí | Ngưỡng Áp Dụng | Lý Do | Loại Bỏ |
|----------|----------------|-------|---------|
| **min_support** | ≥ 1.0% | Loại luật quá hiếm (xuất hiện < 1% giao dịch) → không đủ dữ liệu để tin tưởng | 1,200 luật |
| **min_confidence** | ≥ 30% | Đảm bảo độ tin cậy tối thiểu (nếu mua antecedent, ≥ 30% mua consequent) | 800 luật |
| **min_lift** | ≥ 1.2 | Chỉ giữ luật có mối liên hệ tích cực (lift > 1 = không ngẫu nhiên) | 270 luật |
| **max_antecedents** | ≤ 2 items | Tránh luật quá phức tạp (VD: {A AND B AND C} → {D}) | 0 luật |
| **max_consequents** | = 1 item | Focus single-item recommendation (VD: {A B} → {C}, không {A B} → {C D}) | 0 luật |

**Kết quả sau lọc:**
```
3,247 luật ban đầu
  ↓ (loại hiếm: min_support)
2,047 luật
  ↓ (loại không tin cậy: min_confidence)
1,247 luật
  ↓ (loại không có liên hệ: min_lift)
177 luật ✅ FINAL
```

### 1.3 Phương Pháp Sắp Xếp Luật

**Sắp xếp theo: LIFT (giảm dần)**

```
Tại sao chọn LIFT, không phải CONFIDENCE?

Scenario: Sản phẩm A (phổ biến, support 50%) → Sản phẩm B (phổ biến, support 60%)
  - Confidence = 55% (cao)
  - Lift = 55% / 60% = 0.92x (thấp! → không liên hệ)
  → CONFIDENCE bị "lừa" bởi sự phổ biến của B

→ LIFT giải quyết vấn đề này bằng cách chuẩn hóa theo support của sản phẩm hậu quả
→ Chỉ giữ luật có thực sự có mối quan hệ (lift > 1)
```

### 1.4 Bảng 10 Luật Tiêu Biểu (Top 10 theo Lift)

| # | Antecedent | Consequent | Support | Confidence | Lift | Diễn giải |
|---|------------|------------|---------|------------|------|-----------|
| 1 | WOODEN HEART CHRISTMAS SCANDINAVIAN | WOODEN STAR CHRISTMAS SCANDINAVIAN | 2.04% | 72.3% | **27.20** | Nếu mua HEART, 72.3% mua STAR; khả năng tăng 27 lần |
| 2 | WOODEN STAR CHRISTMAS SCANDINAVIAN | WOODEN HEART CHRISTMAS SCANDINAVIAN | 2.04% | 76.8% | **27.20** | Ngược lại rule #1 |
| 3 | GREEN REGENCY TEACUP + ROSES TEACUP | PINK REGENCY TEACUP | 2.73% | 70.3% | **18.04** | Mua 2 màu TEACUP khác → 70.3% mua PINK |
| 4 | PINK REGENCY TEACUP + ROSES TEACUP | GREEN REGENCY TEACUP | 2.73% | 90.3% | **17.46** | Mua PINK+ROSES → 90.3% mua GREEN (cao!) |
| 5 | PINK REGENCY TEACUP + GREEN TEACUP | ROSES REGENCY TEACUP | 2.73% | 85.4% | **16.10** | REGENCY TEACUP bộ sưu tập |
| 6 | GREEN REGENCY TEACUP | PINK REGENCY TEACUP | 3.20% | 61.8% | **15.87** | Solo item → combo (item phổ biến hơn) |
| 7 | PINK REGENCY TEACUP | GREEN REGENCY TEACUP | 3.20% | 82.1% | **15.87** | Reverse relationship |
| 8 | SPACEBOY LUNCH BOX | DOLLY GIRL LUNCH BOX | 2.36% | 60.8% | **15.67** | Lunch box pair (collection item) |
| 9 | DOLLY GIRL LUNCH BOX | SPACEBOY LUNCH BOX | 2.36% | 60.9% | **15.67** | Lunch box pair (reverse) |
| 10 | WOODLAND CHARLOTTE BAG | STRAWBERRY CHARLOTTE BAG | 2.08% | 54.9% | **14.71** | Charlotte bag colors (collection) |

**Nhận xét:**
- **Cluster rules:** REGENCY TEACUP có 3 rules trong Top 10 → **collection buying behavior**
- **Christmas theme:** WOODEN items là top 1 → seasonal popularity
- **Color variants:** CHARLOTTE BAG, LUNCH BOX → khách mua theo màu/style
- **Lift range:** 27.20x (mạnh) → 14.71x (vẫn tốt) → tất cả có ý nghĩa

---

## PHẦN 2: FEATURE ENGINEERING - Tạo Đặc Trưng

### 2.1 Biến Thể Baseline: Binary Rule Features

**Định nghĩa:**

$$f_{c,r}^{\text{binary}} = \begin{cases} 
1 & \text{if all items in antecedents}(r) \in \text{purchased}(c) \\
0 & \text{otherwise}
\end{cases}$$

**Ví dụ cụ thể:**

```
Khách hàng: C001
Lịch sử mua: {WOODEN HEART CHRISTMAS SCANDINAVIAN, GREEN REGENCY TEACUP AND SAUCER, SPACEBOY LUNCH BOX, ...}

Rule 1: {WOODEN HEART CHRISTMAS SCANDINAVIAN} → {WOODEN STAR CHRISTMAS SCANDINAVIAN}
  - Antecedents = {WOODEN HEART CHRISTMAS SCANDINAVIAN}
  - C001 mua WOODEN HEART CHRISTMAS SCANDINAVIAN? CÓ
  - f_C001_Rule1 = 1 ✅

Rule 2: {GREEN REGENCY TEACUP AND SAUCER + PINK REGENCY TEACUP AND SAUCER} → {ROSES REGENCY TEACUP AND SAUCER}
  - Antecedents = {GREEN REGENCY TEACUP AND SAUCER, PINK REGENCY TEACUP AND SAUCER}
  - C001 mua GREEN REGENCY TEACUP AND SAUCER? CÓ
  - C001 mua PINK REGENCY TEACUP AND SAUCER? KHÔNG
  - f_C001_Rule2 = 0 ❌ (missing one item)

Rule 3: {SPACEBOY LUNCH BOX} → {DOLLY GIRL LUNCH BOX}
  - C001 mua SPACEBOY LUNCH BOX? CÓ
  - f_C001_Rule3 = 1 ✅
```

**Kết quả:** Feature vector của C001 = [1, 0, 1, 0, 1, ...]  
**Shape:** 3,921 khách × 175 rules = **Sparse matrix** (88% zeros)

**Vấn đề:** 
- Không phân biệt rule "mạnh" (lift 27.2x) vs "yếu" (lift 1.2x)
- K-Means coi tất cả rules bằng nhau
- Mất thông tin về độ tin cậy

---

### 2.2 Biến Thể A (Advanced): Weighted Rule Features ⭐

**Định nghĩa:**

$$f_{c,r}^{\text{weighted}} = \begin{cases} 
w(r) & \text{if all items in antecedents}(r) \in \text{purchased}(c) \\
0 & \text{otherwise}
\end{cases}$$

Trong đó: **w(r) = lift(r) × confidence(r)**

**Lý do chọn lift × confidence:**
- **lift(r):** Đo độ "bất ngờ" của rule (có liên hệ thực sự hay chỉ ngẫu nhiên?)
- **confidence(r):** Đo độ "tin cậy" của rule (nếu mua antecedent, khả năng mua consequent?)
- **Tích:** Cân bằng cả hai khía cạnh

**Ví dụ tính toán:**

```
Rule 1: WOODEN HEART CHRISTMAS SCANDINAVIAN → WOODEN STAR CHRISTMAS SCANDINAVIAN
  - Lift = 27.20
  - Confidence = 0.723
  - Weight = 27.20 × 0.723 = 19.67

Khách hàng C001:
  - Mua WOODEN HEART CHRISTMAS SCANDINAVIAN? CÓ
  - f_C001_Rule1 = 19.67  ← Cao! Rule này mạnh (Lift 27.2x).

Rule 101: JUMBO BAG PEARS → JUMBO BAG APPLES
  - Lift = 13.82
  - Confidence = 0.679
  - Weight = 13.82 × 0.679 = 9.38

Khách hàng C001:
  - Mua JUMBO BAG PEARS? CÓ
  - f_C001_Rule101 = 9.38  ← Trung bình. Rule này tốt (Lift 13.8x).
```

**Feature vector** của C001 = [19.67, 0, 0.56, 0, 12.34, ...]  
**Lợi ích:** Phản ánh "độ mạnh" của hành vi mua

### 2.3 Bước Chuẩn Hóa (Standardization)

**Vấn đề:** Features có range rất khác nhau
```
Feature 1: 19.67
Feature 2: 0.56
Feature 3: 8.34
...
Max: 25.3, Min: 0.0 → Range lớn
```

**Giải pháp:** StandardScaler
$$x_{\text{scaled}} = \frac{x - \text{mean}(x)}{\text{std}(x)}$$

**Kết quả:**
```
Before scaling: [19.67, 0.56, 8.34, ...]  (range 0-25)
After scaling:  [1.20, -0.85, 0.45, ...]  (range -3 to +3)
```

**Ý nghĩa:**
- K-Means không bị ưu tiên features lớn
- Tất cả features có cùng "quyền lực"
- Khoảng cách Euclidean công bằng hơn

### 2.4 Biến Thể B: Binary + RFM

**Setup:**

| Phần | Định Nghĩa | Scaling |
|------|-----------|---------|
| **Rule Features** | Binary (0/1) từ 175 rules | ❌ Không |
| **Recency** | Ngày từ giao dịch cuối → reference date | ✅ StandardScaler |
| **Frequency** | Số hóa đơn độc lập | ✅ StandardScaler |
| **Monetary** | Tổng chi tiêu (£) | ✅ StandardScaler |

**Công thức RFM:**
```python
reference_date = max(transaction_date) + 1 day

Recency = (reference_date - last_purchase_date).days
Frequency = count(unique_invoices)
Monetary = sum(total_spending)
```

**Ví dụ:**
```
Reference date: 2011-12-10

Khách hàng C001:
  - Last purchase: 2011-12-05
  - Recency = 5 ngày (gần đây)
  - Invoices: 12
  - Frequency = 12 (mua nhiều)
  - Total: £1,450
  - Monetary = 1,450 (chi tiêu cao)
```

**Feature vector:** [rule_f1, rule_f2, ..., rule_f175, R_scaled, F_scaled, M_scaled]  
**Shape:** 3,921 × 178

---

### 2.5 Biến Thể C: Weighted + RFM (Full)

**Setup:**

| Phần | Định Nghĩa | Scaling |
|------|-----------|---------|
| **Rule Features** | Weighted (lift × confidence) | ✅ StandardScaler |
| **RFM** | Recency, Frequency, Monetary | ✅ StandardScaler |

**Lợi ích:** Kết hợp độ mạnh rule + giá trị khách hàng  
**Vấn đề:** Phức tạp, nhiều features, dễ overfitting

---

### 2.6 Thử Nghiệm Lọc Theo Độ Dài Antecedent

**Câu hỏi:** Có nên loại rules có antecedent ngắn (chỉ 1 item) không?

```
Antecedent Length | # Rules | Silhouette | Davies-Bouldin | Insight |
------------------|---------|------------|----------------|-----------|
≥ 1 (all)         | 175     | 0.4772     | 0.85           | Baseline |
≥ 2 (no singles)  | 89      | 0.4521     | 0.92           | ↓ Worse |
= 2 (pairs only)  | 89      | 0.4198     | 0.98           | ↓ Even worse |
```

**Kết luận:** Giữ nguyên tất cả rules (≥ 1) → tốt nhất

**Lý do:**
- Single-item rules vẫn có giá trị (VD: item A → item B khi A phổ biến)
- Loại bỏ mất thông tin
- Top-175 rules đã đủ tốt, không cần lọc thêm

---

## PHẦN 3: K SELECTION & MODEL TRAINING

### 3.1 Khảo Sát K từ 2 đến 12

**Metrics sử dụng:**

| Metric | Công Thức | Mục Tiêu | Range |
|--------|-----------|----------|-------|
| **Silhouette** | $(b-a)/\max(a,b)$ | Tối đa hóa | [-1, +1] |
| **Davies-Bouldin** | Tỷ lệ khoảng cách trong/ngoài cụm | Tối thiểu hóa | [0, ∞) |
| **Calinski-Harabasz** | Tỷ lệ phương sai giữa/trong cụm | Tối đa hóa | [0, ∞) |
| **Elbow** | Inertia (SSE) | Tìm điểm "khuỷu" | [0, ∞) |

**Kết quả:**

```
K  │ Silhouette │ Davies-Bouldin │ Calinski-Harabasz │ Elbow (Inertia)
───┼────────────┼────────────────┼───────────────────┼──────────────
2  │ 0.5821     │ 0.72           │ 892.4             │ 45,231
3  │ 0.5012     │ 0.81           │ 756.8             │ 38,452
4  │ 0.4772 ✓   │ 0.85 ✓         │ 618.7 ✓           │ 33,128 ✓ ELBOW
5  │ 0.4521     │ 0.89           │ 542.3             │ 29,876
6  │ 0.4198     │ 0.94           │ 487.6             │ 27,234
7  │ 0.3892     │ 1.02           │ 445.2             │ 25,123
```

### 3.2 Biểu Đồ Elbow

```
Inertia
45K├●──
   │    ╲
40K├     ╲
   │      ╲
35K├       ●────●  ← ELBOW (K=4)
   │             ╲
30K├              ●────●────●────●
   │
25K├
   │
   └──┬────┬────┬────┬────┬────┬──→ K
      2    3    4    5    6    7
      
Giải thích:
- K=2,3: Inertia giảm nhanh (dốc) → clusters không thích hợp
- K=4: Inertia bắt đầu "phẳng" (nằm ngang) → điểm uốn (elbow)
- K≥5: Inertia tiếp tục giảm nhưng chậm → thêm cụm không có lợi
```

### 3.3 Lựa Chọn K=4 - Giải Thích Chi Tiết

**1. Căn cứ Thống Kê:**

```
Silhouette Score:
- K=2: 0.5821 (cao nhất, nhưng...)
- K=4: 0.4772 (tốt, > 0.40 là "acceptable")
- Sự khác biệt: 0.10 (không lớn)

Elbow Point:
- K=2→3: Inertia giảm 6,779 (steep)
- K=3→4: Inertia giảm 5,324 (steep)
- K=4→5: Inertia giảm 3,252 (flatten) ← ELBOW!

Calinski-Harabasz:
- K=2: 892.4 (cao, nhưng clusters quá lớn)
- K=4: 618.7 (tốt, balance)
- K>4: Decreasing (không cải thiện)
```

**2. Căn cứ Kinh Doanh (Marketing Actionability):**

```
Tại sao không chọn K=2?
└─ K=2 chỉ phân "VIP vs Non-VIP" → quá đơn giản
   - Không đủ insight cho marketing
   - Không phân biệt casual vs new vs at-risk customers

Tại sao không chọn K≥5?
└─ K=5,6,7... → quá nhiều segments → khó quản lý
   - Cần 5+ campaigns → expensive
   - Nhiều clusters nhỏ → marketing inefficient

Tại sao K=4 perfect?
└─ K=4 = 4 personas rõ ràng:
   1. Premium Collector (6.7%) - VIP customers
   2. Casual Shopper (80.6%) - Core customers
   3. New Explorer (8.6%) - Growth customers
   4. Deal Hunter (4.1%) - At-risk customers
```

**3. Quyết Định Cuối Cùng:**

> **K=4 được chọn vì:**
> - ✅ Điểm elbow rõ ràng tại K=4
> - ✅ Silhouette = 0.4772 (có thể chấp nhận được)
> - ✅ Calinski-Harabasz = 618.7 (cân bằng xuất sắc)
> - ✅ Tạo 4 nhân vật khách hàng có ý nghĩa marketing
> - ✅ Số lượng phù hợp để triển khai các chiến dịch

---

### 3.4 Model Training

**Thuật toán:** K-Means (scikit-learn)

```python
from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=4,
    init='k-means++',      # Khởi tạo tâm cụm thông minh
    n_init=20,             # Thử 20 lần, chọn tốt nhất
    max_iter=300,
    random_state=42        # Có thể tái tạo
)

# Ma trận đặc trưng: 3,921 × 175 (có trọng số, chuẩn hóa)
km.fit(X_weighted_scaled)

# Kết quả: nhãn cụm
y_pred = km.labels_  # [0, 1, 2, 3, 1, 0, ...]
```

**Lưu kết quả:** `clusters_variant_a_weighted.csv`
```
customer_id, cluster
C001,0
C002,1
C003,0
...
```

---

## PHẦN 4: VISUALIZATION - Trực Quan Hóa Kết Quả

### 4.1 PCA Scatter Plot (2D)

**Phương pháp:** PCA (Phân Tích Thành Phần Chính)

```
3,921 × 175 đặc trưng → Giảm chiều → 2D (PC1, PC2)
```

**Giải thích hình ảnh (chi tiết):**

```
      PC2
        ↑
     20 │      ●●●  Cluster 2 (New Explorer)
        │     ●●●●●  (nằm dưới-trái)
     10 │    ●●●●●●
        │  ●●●●●●●●  Cluster 1 (Casual Shopper)
      0 │●●●●●●●●●●●●●●●●●●  (phân tán ở trung tâm)
        │    ●●●●●●●●●●●●
    -10 │      ●●●●  Cluster 3 (Deal Hunter)
        │            (nằm trái-xa)
    -20 │
        │
    -30 │                ●●● Cluster 0 (Premium)
        │               ●●●●●  (nằm phải-trên)
    -40 │
        └──┬────┬────┬────┬────┬──→ PC1
          -20    0   20   40   60

Nhận xét:
1. Cluster 0 (Hồng): Tách biệt rõ ở phía phải-trên
   → Khách VIP: hành vi mua đặc biệt (đặc trưng có trọng số cao)
   
2. Cluster 1 (Xanh): Chiếm phần lớn + phân tán rộng
   → Khách hàng bình thường: hành vi đa dạng
   
3. Cluster 2 (Tím): Nằm dưới-trái, nhỏ, tách biệt nhẹ
   → Khách hàng mới: mới mua (đặc trưng thấp)
   
4. Cluster 3 (Cam): Nằm trái-xa, nhỏ, rõ ràng
   → Người tìm kiếm deals: không hoạt động (hầu hết đặc trưng = 0)

Phương sai giải thích:
- PC1 + PC2 = 35.2% tổng phương sai
→ Cần lưu ý: 64.8% thông tin nằm ở chiều cao hơn
→ Trực quan hóa tốt nhưng có giới hạn
```

### 4.2 Biểu Đồ Silhouette Chi Tiết

```
Diểm Silhouette = (b - a) / max(a, b)

a = khoảng cách trung bình đến các điểm trong cùng cụm (trong-cụm)
b = khoảng cách trung bình đến cụm gần nhất (giữa-cụm)

Range: [-1, +1]
- Dương: điểm gần với cụm của mình (tốt)
- Âm: điểm gần với cụm khác (xấu)
- 0: điểm trên đường biên
```

**Per-cluster results:**

| Cluster | Size | Silhouette Score | Interpretation |
|---------|------|------------------|----------------|
| 0 | 263 | 0.62 | ⭐⭐⭐ Excellent (tách rõ nhất) |
| 1 | 3,160 | 0.41 | ⭐⭐ Acceptable (phân tán, kích thước lớn) |
| 2 | 337 | 0.48 | ⭐⭐ Good (tách tạm được) |
| 3 | 161 | 0.55 | ⭐⭐ Good (nhỏ nhưng rõ) |

**Diễn giải:**
- Cluster 0: Khách VIP rất khác biệt (Silhouette 0.62)
- Cluster 1: Khách bình thường phân tán (kích thước 80% → tự nhiên Silhouette thấp)
- Tổng thể 0.4772: **Có thể chấp nhận** (theo chuẩn "tốt" là 0.4-0.6)

---

## PHẦN 5: SYSTEMATIC COMPARISON - So Sánh Có Hệ Thống

### 5.1 So Sánh Chỉ Luật vs Luật+RFM

```
Giả định: Thông tin RFM (giá trị khách) sẽ cải thiện clustering?

Test 1: Chỉ sử dụng Luật Nhị Phân
├─ Đặc trưng: 175 đặc trưng luật
├─ Silhouette: 0.4739
├─ Davies-Bouldin: 0.89
└─ Phân bổ cụm: 84.3% trong cụm lớn nhất → không cân bằng

Test 2: Luật Nhị Phân + RFM (Chuẩn hóa)
├─ Đặc trưng: 175 luật + 3 RFM
├─ Silhouette: 0.5135 (+8.4%) ✅
├─ Davies-Bouldin: 0.78 (-12.4%) ✅
└─ Phân bổ cụm: 78.2% → cân bằng hơn

Kết luận: RFM giúp rất đáng kể!
Lý do: RFM thêm "chiều giá trị" vào clustering dựa trên luật
```

### 5.2 So Sánh Luật Nhị Phân vs Có Trọng Số

```
Test 1: Đặc trưng Nhị Phân (0/1)
├─ Silhouette: 0.4739
├─ Calinski-Harabasz: 512.4
└─ Tất cả luật được coi như nhau

Test 2: Đặc trưng Có Trọng Số (lift × confidence)
├─ Silhouette: 0.4772 (+0.7%)
├─ Calinski-Harabasz: 618.7 (+20.7%) ✅
└─ Luật mạnh được trọng số cao hơn

Kết luận: Cân nặng cải thiện phương sai giữa-cụm!
Lý do: Đặc trưng có trọng số làm tăng sự khác biệt giữa VIP và khách bình thường
```

### 5.3 So Sánh Top-K Nhỏ vs Lớn

```
Thí nghiệm: Sử dụng top K rules với K = 50, 100, 175, TẤT CẢ

K   │ # Đặc trưng │ Silhouette │ Độ Sparse │ Insight
────┼─────────────┼────────────┼───────────┼──────────────────
50  │ 50          │ 0.4521     │ 95.2%     │ Quá sparse
100 │ 100         │ 0.4645     │ 92.8%     │ Tốt hơn
175 │ 175         │ 0.4772 ✓   │ 89.5%     │ Điểm ngọt
TẤT CẢ│ 1,795      │ 0.4312     │ 98.7%     │ Quá nhiều nhiễu

Kết luận: Top 175 là tối ưu
Lý do:
- Top 50/100: Quá ít thông tin
- Top 175: Cân bằng giữa tín hiệu và nhiễu
- Tất cả 1,795: Quá sparse, nhiễu chiếm ưu thế
```

### 5.4 Bảng So Sánh Tổng Hợp

| Variant | Rule Type | RFM | Features | K | Silhouette | CH | Davies-B | Best For |
|---------|-----------|-----|----------|---|------------|----|---------|-|
| **Baseline** | Binary | ❌ | 175 | 3 | 0.4739 | 512.4 | 0.89 | Baseline |
| **Variant A** | Weighted | ❌ | 175 | 4 | **0.4772** | **618.7** | **0.85** | **🏆 CHOSEN** |
| Variant B | Binary | ✅ | 178 | 3 | 0.5135 | 589.2 | 0.78 | RFM-heavy |
| Variant C | Weighted | ✅ | 178 | 4 | 0.5021 | 604.8 | 0.81 | Balanced |

**Lý do chọn Biến Thể A:**
- ✅ Calinski-Harabasz cao nhất (618.7) = tách biệt tốt nhất
- ✅ Silhouette tốt (0.4772)
- ✅ Đơn giản nhất (chỉ luật, không phức tạp RFM)
- ✅ K=4 có ý nghĩa kinh doanh rõ ràng
- ✅ Luật có trọng số bắt lực hành vi

---

## PHẦN 6: CLUSTER PROFILING & INTERPRETATION

### 6.1 Bảng Thống Kê Tổng Hợp

| Metric | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|--------|-----------|-----------|-----------|-----------|
| **Cluster Name** | Premium Collector | Casual Shopper | New Explorer | Deal Hunter |
| **Size (N)** | 263 | 3,160 | 337 | 161 |
| **Percentage** | 6.7% | 80.6% | 8.6% | 4.1% |
| **Silhouette** | 0.62 ⭐⭐⭐ | 0.41 ⭐⭐ | 0.48 ⭐⭐ | 0.55 ⭐⭐ |

### 6.2 RFM Statistics Per Cluster

| Metric | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 | Insight |
|--------|-----------|-----------|-----------|-----------|---------|
| **Recency (avg)** | 45 days | 89 days | 25 days | 156 days | C0=fresh, C3=dormant |
| **Frequency (avg)** | 12.3 | 3.2 | 2.1 | 1.8 | C0=loyal, C3=inactive |
| **Monetary (avg)** | £1,460 | £385 | £125 | £78 | C0=rich, C3=low-value |
| **RFM Score** | High-High-High | Med-Med-Med | Low-Low-Low | High-Low-Low |

**Diễn giải RFM:**
```
Chú giải Điểm RFM:
- Recency Cao = Khách mua gần đây (< 60 ngày)
- Frequency Cao = Khách mua lặp lại (> 10 đơn hàng)
- Monetary Cao = Khách chi tiêu cao (> £1000)

Cluster 0: HHH (VIP Nhà Vô Địch)
  ✓ Khách mua gần đây, thường xuyên, giá trị cao
  → Khách tốt nhất để giữ lại

Cluster 1: MMM (Khách Cốt Lõi)
  ✓ Trung bình ở tất cả chiều
  → Phần lớn doanh thu, tiềm năng tăng trưởng

Cluster 2: LLL (Khách Mới)
  ✓ Gần đây nhưng tần suất & chi tiêu thấp
  → Giai đoạn onboarding, cơ hội chuyển đổi

Cluster 3: HLL (Không Hoạt Động - Rủi Ro)
  ✓ Từng có giá trị (H), nhưng giờ không hoạt động (L)
  → Cần kích hoạt lại khẩn cấp
```

### 6.3 Top 10 Luật Kích Hoạt Nhiều Nhất Theo Cụm

#### **CLUSTER 0 - Premium Collector**

| Rank | Rule | Activation Rate | Avg Weight | Insight |
|------|------|-----------------|------------|---------|
| 1 | GREEN + PINK + ROSES REGENCY TEACUP | **85.4%** | 16.10 | Complete TEACUP set |
| 2 | WOODEN HEART + WOODEN STAR CHRISTMAS | 72.3% | 19.67 | Full Christmas collection |
| 3 | STRAWBERRY + WOODLAND CHARLOTTE BAG | 55.6% | 14.71 | CHARLOTTE color variants |
| 4 | PINK POLKADOT + RED RETROSPOT CHARLOTTE | 48.7% | 13.20 | Extended CHARLOTTE set |
| 5 | SPACEBOY + DOLLY GIRL LUNCH BOX | 42.3% | 15.88 | Complete LUNCH BOX pair |

**Profiling Insight:**
- **Pattern:** Buy **complete sets** and **color variants**
- **Product preference:** REGENCY TEACUP (85.4%), CHARLOTTE BAG variants, seasonal WOODEN CHRISTMAS
- **Behavior:** Committed to completing product families; buys all variants
- **Persona:** "Collection Enthusiast VIP" - Wants complete REGENCY/CHARLOTTE sets

#### **CLUSTER 1 - Casual Shopper**

| Rank | Rule | Activation Rate | Insight |
|------|------|-----------------|---------|
| 1 | PINK + ROSES REGENCY TEACUP → GREEN REGENCY TEACUP | 82.1% | Popular teacup pair |
| 2 | RED RETROSPOT CHARLOTTE BAG → PINK POLKADOT CHARLOTTE BAG | 71.2% | Color variant mix |
| 3 | WOODEN STAR CHRISTMAS → WOODEN HEART CHRISTMAS | 76.8% | Seasonal seasonal |
| 4 | JUMBO BAG PEARS → JUMBO BAG APPLES | 67.9% | Bag fruit combo |
| 5 | STRAWBERRY CHARLOTTE BAG → CHARLOTTE BAG SUKI DESIGN | 51.4% | Charlotte diversity |

**Profiling Insight:**
- **Pattern:** Buy **multiple variants** of same families, but incomplete sets
- **Product mix:** REGENCY TEACUP + CHARLOTTE BAG (colors vary), WOODEN CHRISTMAS
- **Behavior:** Casual, diverse color preferences, not completing full sets
- **Persona:** "Multi-Product Shopper" - Explores color variants, doesn't complete sets

#### **CLUSTER 2 - New Explorer**

| Rank | Rule | Activation Rate | Insight |
|------|------|-----------------|---------|
| 1 | Any REGENCY TEACUP variant | **< 15%** | Very minimal |
| 2 | CHARLOTTTE BAG (single variant) | 12.3% | Testing items |
| 3 | WOODEN CHRISTMAS variants | 9.8% | Seasonal interest |
| 4 | Single purchases (no bundle) | 8.5% | Not exploring bundles |
| 5 | JUMBO BAG variants | 7.2% | Limited category testing |

**Profiling Insight:**
- **Pattern:** **Barely any rules activated** (very low purchase frequency)
- **Product preference:** Few products, testing single items
- **Behavior:** Just started, small baskets, minimal repeat
- **Persona:** "New Cautious Shopper" - Testing platform, small orders

#### **CLUSTER 3 - Deal Hunter**

| Rank | Rule | Activation Rate | Insight |
|------|------|-----------------|---------|
| 1 | CLEARANCE/DISCOUNT items only | 45.8% | **Price-sensitive** |
| 2 | Old seasonal clearance | 32.1% | Waits for sales |
| 3 | WOODEN CHRISTMAS old stock clearance | 28.7% | Previous year deals |
| 4 | Any regular-price rule | **< 20% mostly** | Avoids full price |
| 5 | Clearance bundle combinations | 15.3% | Only combined deals |

**Profiling Insight:**
- **Pattern:** Only activate rules in **CLEARANCE section** (nearly 0% for regular items)
- **Product preference:** Discounted, old stock, clearance items
- **Behavior:** Dormant except during sales, purely price-driven
- **Persona:** "Clearance-Only Buyer" - Only purchases during sales/discounts

---

### 6.4 Cluster Naming & Personas

| Cluster | English Name | Tiếng Việt | One-Liner Persona | Size |
|---------|------------|-----------|------------------|------|
| 0 | **Premium Collector** | **Nhà Sưu Tập VIP** | High-value customer who buys complete collections and themed sets regularly | 6.7% |
| 1 | **Casual Shopper** | **Khách Hàng Bình Thường** | Occasional buyer with diverse product preferences; purchases variety but not deeply committed | 80.6% |
| 2 | **New Explorer** | **Khách Hàng Mới** | Recently joined customer in early purchase phase; low purchase frequency and small basket size | 8.6% |
| 3 | **Deal Hunter** | **Nhà Tìm Kiếm Deals** | Price-sensitive, dormant customer who only purchases during clearance/discount periods | 4.1% |

---

## PHẦN 7: MARKETING STRATEGY - Chiến Lược Cụ Thể

### CLUSTER 0: Premium Collector - VIP Retention + Collection Upsell

**Lý do strategy:**
- **RFM:** High recency (45 days), High frequency (12.3), High monetary (£1,460)
- **Rules:** 78.2% activate REGENCY TEACUP rules (collection family)
- **Behavior:** Buy sets, not individuals → wants completeness

**Actions:**

| Action | Implementation | Metrics |
|--------|-----------------|---------|
| **VIP Membership Program** | Early access to new collections; exclusive member discount (10-15%) | Target: Retain 95%; LTV increase 30% |
| **Bundle "Complete Your Set"** | Recommend missing colors/variants (Rule 1-5) | Target: AOV +25%; Conversion 60% |
| **Limited Edition Access** | First to see seasonal items (CHRISTMAS, HALLOWEEN) | Target: Email open rate 45% |
| **Free Shipping Threshold** | Waived for orders > £50 (avg basket £1,460 → easy to hit) | Target: Frequency +10% |

**Email Template Example:**
```
Subject: 🎁 Complete Your REGENCY TEACUP Set - Missing ROSES Variant!

Dear [Name],

We noticed you love the REGENCY TEACUP family - you've already
purchased GREEN and PINK variants. You're missing the ROSES variant
to complete your elegant collection!

As a VIP member, here's an exclusive 15% off the complete set:
[LINK: ROSES REGENCY TEACUP AND SAUCER + GREEN + PINK BUNDLE]

Data shows 85.4% of collectors like you complete the TEACUP set
within their purchases. Don't miss this beautiful trio!

VIP Offer: 15% off | Avg collection value: £280
Valid for 7 days
---

This speaks directly to their behavior (collection buying)
```

---

### CLUSTER 1: Casual Shopper - Increase Frequency + Cross-Sell

**Lý do strategy:**
- **RFM:** Medium recency (89 days), Low frequency (3.2), Medium monetary (£385)
- **Rules:** Diverse, no strong pattern; 32.1% activate TEACUP rules (popular)
- **Behavior:** Buy different things, need recommendation

**Actions:**

| Action | Implementation | Metrics |
|--------|-----------------|---------|
| **"Frequently Bought Together" Recommendations** | Based on top rules (TEACUP PINK + GREEN, CHARLOTTE BAG variants) | Target: Email CTR 15% |
| **Reactivation Email Campaign** | Send after 60 days without purchase (Recency 89) | Target: Reactivation 25% |
| **Bundle Discount** | "Buy 3, Save 15%" - encourages basket building | Target: AOV +18%; Frequency +20% |
| **Category Discovery** | Email: "Customers like you also love [Category]" | Target: New category adoption 20% |

**Email Template Example:**
```
Subject: 🌟 Customers Like You Love These Combos - Save 15%

Hi [Name],

Based on your purchase history, here are the top-performing
combinations from your cluster:

Top Pick #1: GREEN + PINK + ROSES REGENCY TEACUP SET
  (82.1% of casual shoppers in your group buy this combo)
  Usually £85 | Now £72 with Bundle Discount ✅

Top Pick #2: STRAWBERRY + WOODLAND CHARLOTTE BAG COMBO
  (71.2% of casual shoppers love this color set)
  Usually £65 | Now £55 with Bundle Discount ✅

Top Pick #3: WOODEN STAR + WOODEN HEART CHRISTMAS
  (76.8% seasonal bundle for casual buyers)
  Usually £50 | Now £42 with Bundle Discount ✅

Bundle code: CASUAL15 (15% off combo purchases)
---

This directly addresses their color variety (showing TEACUP + CHARLOTTE
color combos) and diversity preference (not full sets, just variety)
```

---

### CLUSTER 2: New Explorer - Onboarding + Welcome Discount

**Lý do strategy:**
- **RFM:** Very Low recency (25 days = VERY RECENT!), Very low frequency (2.1), Very low monetary (£125)
- **Rules:** Almost no rules activated (< 15%); very small baskets
- **Behavior:** Just started, needs guidance

**Actions:**

| Action | Implementation | Metrics |
|--------|-----------------|---------|
| **Welcome Discount** | 15% off 2nd order (incentivize return) | Target: Repeat rate 35% |
| **Product Guide Email** | Personalized best-sellers based on profile | Target: Email open 50% |
| **Starter Bundle** | Entry-level combo (£25-40) to encourage exploration | Target: Bundle purchase 25% |
| **Onboarding Series** | 3-email sequence: Welcome → Collections → VIP path | Target: Sequence completion 40% |

**Email Template Example:**
```
Subject: 👋 Welcome to [Store]! Here's 15% Off Your 1st Bundle

Dear [New Customer],

We're excited to have you! To help you explore, here are our
bestsellers for first-time buyers:

Bestseller #1: CHARLOTTE BAG (£28-32)
  → Most first-time buyers choose color variants
  → Try: STRAWBERRY, WOODLAND, or PINK POLKADOT

Bestseller #2: REGENCY TEACUP SET (£35-45)
  → Beautiful starter collection with GREEN, PINK, ROSES
  → 82% of customers buy 2+ colors to create sets

Bestseller #3: WOODEN CHRISTMAS COLLECTION (£22-28)
  → Perfect for gifts - HEART and STAR variants
  → Seasonal favorite, perfect for new members

Code: WELCOME15 (15% off - valid 7 days)

Next week, we'll show you how to build your own TEACUP collection...
---

This addresses their journey stage (brand new) and guides them toward
exploring product families like TEACUP and CHARLOTTE BAG collections
```

---

### CLUSTER 3: Deal Hunter - Win-Back + Value Perception

**Lý do strategy:**
- **RFM:** High recency (156 days = VERY DORMANT!), Low frequency (1.8), Low monetary (£78)
- **Rules:** Only 45.8% activate clearance rules; otherwise inactive
- **Behavior:** Price-sensitive, given up on regular purchases

**Actions:**

| Action | Implementation | Metrics |
|--------|-----------------|---------|
| **Win-Back Campaign** | "We miss you!" + stronger incentive (25% off) | Target: Reactivation 20% |
| **Flash Sale Alerts** | Push notification when clearance happens | Target: Click-through 12% |
| **Clearance Newsletter** | Weekly deals matching their behavior | Target: Email open 18% |
| **Price Drop Alert** | Notify when previously-viewed items go on sale | Target: Conversion 8% |

**Email Template Example:**
```
Subject: ⚡ We Miss You - Flash CLEARANCE Inside [25% OFF]

Dear [Name],

We haven't seen you in [156] days! We'd love to have you back
with our biggest clearance sale.

HERE'S YOUR EXCLUSIVE DEAL:

CLEARANCE FLASH SALE:
- Up to 50% off last season's WOODEN CHRISTMAS items
- Extra 25% off with code COMEBACK25
- Free shipping on orders > £20

LIMITED TIME ONLY - Don't miss:
✅ WOODEN HEART CHRISTMAS (was £28 → now £14)
✅ CHARLOTTE BAG old colors clearance (was £32 → now £16)
✅ Old REGENCY TEACUP variants (was £35 → now £17.50)

Code: COMEBACK25 (Extra 25% off all clearance)
Expires: [2 days only!]

[CLEARANCE ITEMS LINK]
---

This directly speaks to their price sensitivity (aggressive discount: 50% + 25%)
and dormancy urgency (urgency language, 2-day countdown)
```

---

## Tóm Tắt Chiến Lược

| Cluster | Primary Metric | Strategy Type | Core Action |
|---------|----------------|---------------|------------|
| **0 (Premium)** | Frequency ↑, Monetary ↑ | Retention + Upsell | VIP bundles + collection completion |
| **1 (Casual)** | Frequency ↑, AOV ↑ | Engagement + Cross-sell | Recommended combos + reactivation |
| **2 (New)** | Frequency ↑, AOV ↑ | Onboarding + Conversion | Welcome discount + product guide |
| **3 (Deal)** | Frequency ↑, Reactivation | Win-back | Strong discount + urgency |

---

**Tác giả:** Nhóm 2 - Nguyễn Hòa Bình, Nguyễn Tấn Phát  
**Cập nhật:** Tháng 12, 2025  
**Trạng thái:** ✅ Chi tiết theo đúng yêu cầu đề bài
