# Feature Engineering cho Phân Cụm Khách Hàng

## Tổng Quan

Notebook `02_feature_engineering_for_clustering.ipynb` thực hiện việc xây dựng các biến thể đặc trưng (feature variants) từ luật kết hợp và RFM để phục vụ cho phân cụm khách hàng.

## Các Biến Thể Đặc Trưng

### 1. Baseline - Binary Rule Features ⭐

**Mô tả**: Đặc trưng nhị phân cơ bản nhất

**Cách thức**:
- Mỗi chiều (feature) tương ứng với một luật kết hợp
- Giá trị = 1 nếu khách hàng đã mua **tất cả** sản phẩm trong antecedents
- Giá trị = 0 nếu không

**Công thức**:
```
feature_j(customer_i) = 1 if customer_i bought all items in antecedents(rule_j)
                        0 otherwise
```

**Cấu hình**:
- Weighting: ❌ Không
- RFM: ❌ Không
- Scale Rules: ❌ Không
- Scale RFM: N/A

**Ưu điểm**:
- Đơn giản, dễ hiểu
- Tính toán nhanh
- Baseline để so sánh

**Nhược điểm**:
- Không phản ánh độ mạnh của luật
- Không có thông tin về giá trị khách hàng
- Tất cả luật được đối xử như nhau

---

### 2. Variant A - Weighted Rule Features 🎯

**Mô tả**: Đặc trưng có trọng số theo độ mạnh của luật

**Cách thức**:
- Tương tự binary nhưng có trọng số
- Trọng số phản ánh độ mạnh và tin cậy của luật

**Công thức**:
```
weight_j = lift_j × confidence_j

feature_j(customer_i) = weight_j if customer_i bought all items in antecedents(rule_j)
                        0         otherwise
```

**Các công thức trọng số có thể thử**:
1. `lift` (chỉ độ mạnh quan hệ)
2. `lift × confidence` ⭐ (khuyến nghị)
3. `lift × support` (quan hệ mạnh + phổ biến)
4. `lift × confidence × support` (tổng hợp đầy đủ)

**Cấu hình**:
- Weighting: ✅ Có (lift × confidence)
- RFM: ❌ Không
- Scale Rules: ❌ Không (giữ nguyên weighted values)
- Scale RFM: N/A

**Ưu điểm**:
- Phản ánh độ quan trọng của mỗi luật
- Luật mạnh hơn đóng góp nhiều hơn vào profile khách hàng
- Vẫn giữ tính sparse

**Nhược điểm**:
- Phức tạp hơn baseline một chút
- Cần chọn công thức weighting phù hợp

---

### 3. Variant B - Binary Rules + RFM 💰

**Mô tả**: Kết hợp pattern mua hàng với giá trị khách hàng

**Cách thức**:
- Ghép binary rule features với RFM features
- RFM được chuẩn hóa (StandardScaler)
- Rule features giữ nguyên (0/1)

**Công thức**:
```
RFM features:
  - Recency: số ngày kể từ lần mua cuối
  - Frequency: số lần mua (số InvoiceNo duy nhất)
  - Monetary: tổng giá trị mua hàng

Combined features = [Binary Rules | RFM_scaled]
```

**Cấu hình**:
- Weighting: ❌ Không
- RFM: ✅ Có (Recency, Frequency, Monetary)
- Scale Rules: ❌ Không (giữ binary 0/1)
- Scale RFM: ✅ Có (StandardScaler) - **bắt buộc**

**Lý do scale RFM**:
- Recency, Frequency, Monetary có thang đo khác nhau rất nhiều
  - Recency: 1-400 ngày
  - Frequency: 1-200 lần
  - Monetary: 100-100000 GBP
- Không scale → RFM sẽ chiếm ưu thế quá mức

**Ưu điểm**:
- Bổ sung thông tin về giá trị khách hàng
- Phân biệt khách hàng VIP vs khách hàng thường
- Tổng hợp cả behavior pattern và value

**Nhược điểm**:
- Tăng số chiều (dimension)
- RFM có thể "át" rule features nếu không scale cẩn thận

---

### 4. Variant C - Weighted Rules + RFM 🚀

**Mô tả**: Biến thể tổng hợp đầy đủ nhất

**Cách thức**:
- Kết hợp weighted rule features với RFM
- **Cả hai phần đều được chuẩn hóa**

**Công thức**:
```
Weighted Rules: feature_j = lift_j × confidence_j (if satisfied)

Combined features = [Weighted Rules_scaled | RFM_scaled]
```

**Cấu hình**:
- Weighting: ✅ Có (lift × confidence)
- RFM: ✅ Có
- Scale Rules: ✅ Có (StandardScaler) - **quan trọng**
- Scale RFM: ✅ Có (StandardScaler)

**Lý do scale weighted rules**:
- Weighted values có thể rất lớn (lift × confidence có thể > 20)
- Không scale → weighted rules sẽ át RFM
- Scale giúp cân bằng contribution giữa rules và RFM

**Ưu điểm**:
- Tổng hợp đầy đủ nhất: độ mạnh luật + giá trị khách hàng
- Cân bằng tốt giữa các loại features
- Tiềm năng cho kết quả tốt nhất

**Nhược điểm**:
- Phức tạp nhất
- Cần hiểu rõ về scaling

---

## So Sánh Các Biến Thể

| Biến Thể | Weighting | RFM | Scale Rules | Scale RFM | Complexity | Use Case |
|----------|-----------|-----|-------------|-----------|------------|----------|
| Baseline | ❌ | ❌ | ❌ | N/A | ⭐ | Baseline comparison |
| Variant A | ✅ | ❌ | ❌ | N/A | ⭐⭐ | Focus on rule strength |
| Variant B | ❌ | ✅ | ❌ | ✅ | ⭐⭐⭐ | Balance pattern + value |
| Variant C | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐ | Most comprehensive |

---

## Tính Năng Nâng Cao

### Lọc Luật Theo Độ Dài Antecedent

**Mục đích**: Loại bỏ luật quá đơn giản

**Cách thức**:
```python
MIN_ANTECEDENT_LENGTH = 2  # Chỉ giữ luật có ít nhất 2 items

rules_filtered = rules[
    rules['antecedent_length'] >= MIN_ANTECEDENT_LENGTH
]
```

**Lý do**:
- Luật có antecedent = 1 item quá đơn giản (A → B)
- Luật có antecedent ≥ 2 items phức tạp hơn, có ý nghĩa hơn
- Ví dụ: `{Teacup A, Teacup B} → Teacup C` thú vị hơn `{Teacup A} → Teacup B`

**Thử nghiệm**:
- MIN_ANTECEDENT_LENGTH = 1: giữ tất cả
- MIN_ANTECEDENT_LENGTH = 2: chỉ luật phức tạp
- So sánh chất lượng cụm

---

## Quy Trình Feature Engineering

### Bước 1: Chuẩn Bị Dữ Liệu

```python
# Load dữ liệu
df_clean = pd.read_csv("data/processed/cleaned_uk_data.csv")
rules = pd.read_csv("data/processed/rules_apriori_top200_selected.csv")

# Tạo Customer × Item matrix
customer_item_bool = create_customer_item_matrix(df_clean)

# Tính RFM
rfm = calculate_rfm(df_clean)
```

### Bước 2: Lọc Luật (Optional)

```python
# Lọc theo độ dài antecedent
MIN_ANTECEDENT_LENGTH = 1  # hoặc 2
rules_filtered = rules[
    rules['antecedent_length'] >= MIN_ANTECEDENT_LENGTH
]
```

### Bước 3: Tạo Features

```python
# Baseline
features_baseline = create_binary_rule_features(
    customer_item_bool, 
    rules_filtered
)

# Variant A
features_weighted = create_weighted_rule_features(
    customer_item_bool, 
    rules_filtered,
    weight_formula='lift_confidence'
)

# Variant B
features_binary_rfm = combine_rules_and_rfm(
    features_baseline, 
    rfm,
    scale_rules=False,
    scale_rfm=True
)

# Variant C
features_weighted_rfm = combine_rules_and_rfm(
    features_weighted, 
    rfm,
    scale_rules=True,
    scale_rfm=True
)
```

### Bước 4: Lưu Features

```python
# Lưu từng biến thể
output_dir = Path("data/features")
features_baseline.to_csv(output_dir / "baseline_binary.csv")
features_weighted.to_csv(output_dir / "variant_a_weighted.csv")
# ...
```

---

## Phân Tích Features

### Thống Kê Cơ Bản

```python
# Độ sparse
sparsity = (features == 0).sum() / features.size * 100

# Giá trị trung bình
mean_value = features.mean()

# Phân bố
plt.hist(features.values.flatten(), bins=50)
```

### PCA Visualization

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)

plt.scatter(features_pca[:, 0], features_pca[:, 1])
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
```

---

## Best Practices

### 1. Scaling Strategy

**Nguyên tắc chung**:
- Binary features (0/1): **Không cần scale**
- Weighted features (giá trị lớn): **Nên scale** nếu kết hợp với features khác
- RFM: **Luôn scale** (bắt buộc)

**Khi nào scale**:
- Khi kết hợp nhiều loại features với thang đo khác nhau
- Khi sử dụng thuật toán dựa trên khoảng cách (K-Means, DBSCAN)
- Khi muốn cân bằng contribution giữa các feature groups

**Khi nào không scale**:
- Khi chỉ dùng binary features
- Khi muốn giữ tính interpretability
- Khi test baseline

### 2. Weighting Formula Selection

**Lift only**:
- Ưu tiên độ mạnh quan hệ
- Không quan tâm tần suất

**Lift × Confidence** ⭐:
- Cân bằng độ mạnh và độ tin cậy
- Khuyến nghị cho hầu hết trường hợp

**Lift × Support**:
- Ưu tiên luật phổ biến
- Tránh overfit vào luật hiếm

**Lift × Confidence × Support**:
- Tổng hợp đầy đủ
- Có thể quá conservative

### 3. RFM Integration

**Best practices**:
- Luôn scale RFM (StandardScaler)
- Xem xét log transform cho Monetary (nếu skewed)
- Kiểm tra correlation giữa R, F, M

**Common mistakes**:
- ❌ Không scale RFM → RFM át rule features
- ❌ Scale rules nhưng không scale RFM → mất cân bằng
- ❌ Dùng RFM raw values → dominated by Monetary

---

## Ví Dụ Cụ Thể

### Case Study: Khách Hàng A

**Profile**:
- Đã mua: {Teacup Pink, Teacup Green, Teacup Roses}
- RFM: Recency=10 days, Frequency=15, Monetary=500 GBP

**Rules**:
1. {Teacup Pink, Teacup Green} → Teacup Roses (lift=18, conf=0.7)
2. {Teacup Pink} → Teacup Green (lift=15, conf=0.8)

**Features**:

| Biến Thể | Rule 1 | Rule 2 | RFM_R | RFM_F | RFM_M |
|----------|--------|--------|-------|-------|-------|
| Baseline | 1 | 1 | - | - | - |
| Variant A | 12.6 | 12.0 | - | - | - |
| Variant B | 1 | 1 | -1.2 | 0.8 | 0.5 |
| Variant C | 0.9 | 0.8 | -1.2 | 0.8 | 0.5 |

*Giá trị scaled là ví dụ minh họa*

---

## Đánh Giá Chất Lượng Features

### Metrics

1. **Sparsity**: % features = 0
   - Quá sparse (>95%) → features không đủ discriminative
   - Quá dense (<50%) → features có thể không meaningful

2. **Variance**: Độ phân tán
   - Variance cao → features đa dạng
   - Variance thấp → features không phân biệt

3. **Separation**: PCA visualization
   - Clusters rõ ràng → features tốt
   - Overlapping nhiều → features cần cải thiện

### So Sánh Giữa Các Biến Thể

Sử dụng clustering metrics (sẽ làm ở notebook tiếp theo):
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

---

## Bước Tiếp Theo

Sau khi có các biến thể features:

1. **Clustering**: Áp dụng K-Means trên từng biến thể
2. **Evaluation**: So sánh chất lượng cụm
3. **Analysis**: Giải thích đặc điểm từng cụm
4. **Selection**: Chọn biến thể tốt nhất

Xem notebook: `03_clustering_and_evaluation.ipynb`

---

## Tham Khảo

- Agrawal, R., & Srikant, R. (1994). "Fast Algorithms for Mining Association Rules"
- Hughes, A. M. (1994). "Strategic Database Marketing"
- Tan, P. N., et al. (2005). "Introduction to Data Mining" - Chapter 8: Cluster Analysis
