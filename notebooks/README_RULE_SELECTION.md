# Quy Trình Chọn Lọc Luật Kết Hợp cho Phân Cụm Khách Hàng

## Tổng Quan

Notebook `01_rule_selection_for_clustering.ipynb` trình bày chi tiết quy trình chọn lọc luật kết hợp (Association Rules) từ thuật toán Apriori hoặc FP-Growth để làm đầu vào cho bước phân cụm khách hàng.

## Mục Tiêu

1. **Phân tích chất lượng luật**: So sánh và đánh giá các luật từ Apriori và FP-Growth
2. **Xác định tiêu chí lọc**: Chọn ngưỡng và số lượng luật phù hợp
3. **Trích xuất luật tiêu biểu**: Hiển thị 10 luật hàng đầu với các chỉ số quan trọng
4. **Giải thích lý do chọn**: Làm rõ cơ sở khoa học cho việc lựa chọn

## Tiêu Chí Chọn Lọc

### 1. Thuật Toán
- **Apriori** hoặc **FP-Growth** (có thể chọn dựa trên kết quả so sánh)
- Cả hai thuật toán cho kết quả giống nhau về mặt luật, nhưng khác về hiệu suất

### 2. Các Ngưỡng Lọc

| Chỉ số | Ngưỡng | Lý do |
|--------|--------|-------|
| **Support** | ≥ 0.01 (1%) | Đảm bảo luật xuất hiện đủ phổ biến trong dữ liệu |
| **Confidence** | ≥ 0.3 (30%) | Độ tin cậy hợp lý, không quá thấp |
| **Lift** | ≥ 1.2 | Chỉ giữ luật có mối quan hệ dương và đủ mạnh |

### 3. Top-K Rules
- **K = 200 luật** (có thể điều chỉnh)
- Sắp xếp theo **Lift** giảm dần
- Lý do: Cung cấp đủ đa dạng nhưng không quá phức tạp

## Giải Thích Các Chỉ Số

### Support (Độ Hỗ Trợ)
```
Support(A → B) = P(A ∩ B) = (Số giao dịch chứa cả A và B) / (Tổng số giao dịch)
```
- **Ý nghĩa**: Tỷ lệ giao dịch chứa cả antecedent và consequent
- **Cao**: Luật phổ biến
- **Thấp**: Luật hiếm, có thể là niche

### Confidence (Độ Tin Cậy)
```
Confidence(A → B) = P(B|A) = Support(A ∩ B) / Support(A)
```
- **Ý nghĩa**: Xác suất mua B khi đã mua A
- **Cao**: Dự đoán tốt
- **Thấp**: Mối quan hệ yếu

### Lift (Độ Nâng)
```
Lift(A → B) = P(B|A) / P(B) = Confidence(A → B) / Support(B)
```
- **Ý nghĩa**: Mức độ mạnh mẽ của mối quan hệ
- **Lift > 1**: Quan hệ dương (A làm tăng khả năng mua B)
- **Lift = 1**: Độc lập
- **Lift < 1**: Quan hệ âm (A làm giảm khả năng mua B)

## Ví Dụ Luật Tiêu Biểu

Dưới đây là ví dụ 3 luật từ dữ liệu thực tế:

| Antecedent | Consequent | Support | Confidence | Lift |
|------------|------------|---------|------------|------|
| WOODEN HEART CHRISTMAS SCANDINAVIAN | WOODEN STAR CHRISTMAS SCANDINAVIAN | 0.0204 | 0.7230 | 27.20 |
| GREEN REGENCY TEACUP, ROSES REGENCY TEACUP | PINK REGENCY TEACUP | 0.0273 | 0.7029 | 18.04 |
| SPACEBOY LUNCH BOX | DOLLY GIRL LUNCH BOX | 0.0236 | 0.6077 | 15.67 |

### Giải thích:
1. **Luật 1**: Nếu khách mua "WOODEN HEART CHRISTMAS", có 72.3% khả năng mua "WOODEN STAR CHRISTMAS" (Lift = 27.2 rất cao → quan hệ cực mạnh)
2. **Luật 2**: Bộ tách trà (teacup) có xu hướng mua cùng nhau
3. **Luật 3**: Hộp cơm trưa (lunch box) có pattern mua theo cặp

## Quy Trình Sử Dụng

### Bước 1: Chuẩn Bị Dữ Liệu
```bash
# Đảm bảo đã chạy pipeline tạo luật
python run_papermill.py
```

Điều này sẽ tạo ra:
- `data/processed/rules_apriori_filtered.csv`
- `data/processed/rules_fpgrowth_filtered.csv`

### Bước 2: Chạy Notebook Chọn Luật
```bash
jupyter notebook notebooks/01_rule_selection_for_clustering.ipynb
```

Hoặc sử dụng VS Code để mở và chạy từng cell.

### Bước 3: Điều Chỉnh Tham Số
Trong notebook, có thể thay đổi:
```python
TOP_K = 200  # Số luật muốn chọn
MIN_SUPPORT = 0.01  # Ngưỡng support tối thiểu
MIN_CONFIDENCE = 0.3  # Ngưỡng confidence tối thiểu
MIN_LIFT = 1.2  # Ngưỡng lift tối thiểu
ALGORITHM = "apriori"  # hoặc "fpgrowth"
```

### Bước 4: Xuất Kết Quả
Notebook sẽ tự động xuất file:
```
data/processed/rules_apriori_top200_selected.csv
```

## Kết Quả Mong Đợi

### Output Files
1. **rules_[algorithm]_top[K]_selected.csv**: Bộ luật đã chọn (200 luật)
2. **Visualizations**: Các biểu đồ phân tích phân bố chỉ số
3. **Summary Statistics**: Thống kê mô tả các chỉ số

### Insights
- Top 10 luật với Lift cao nhất
- Phân bố Support, Confidence, Lift của bộ luật
- Độ phủ: số sản phẩm duy nhất trong bộ luật
- So sánh Apriori vs FP-Growth (nếu có)

## Lý Do Chọn Lift Làm Tiêu Chí Chính

### 1. Đo Lường Mối Quan Hệ Phi Ngẫu Nhiên
- Lift > 1: Sản phẩm có xu hướng được mua cùng nhau
- Càng cao càng mạnh mẽ
- Loại bỏ được các luật xảy ra ngẫu nhiên

### 2. Cân Bằng Support và Confidence
- Support cao + Confidence cao ≠ Luật tốt nếu sản phẩm B vốn rất phổ biến
- Lift điều chỉnh để tính đến tần suất cơ bản của consequent

### 3. Phù Hợp Cho Phân Cụm
- Luật có Lift cao tạo ra các pattern rõ ràng
- Giúp phân biệt các nhóm khách hàng khác nhau
- Feature space có ý nghĩa hơn

## Bước Tiếp Theo

Sau khi có bộ luật được chọn, sử dụng cho:

1. **Tạo Feature Vector**: Mỗi khách hàng → vector binary (đã mua đủ antecedent của luật j hay không)
2. **Phân Cụm K-Means**: Sử dụng feature vector để nhóm khách hàng
3. **Phân Tích Cụm**: Giải thích đặc điểm từng nhóm dựa trên luật

Xem notebook tiếp theo: `02_customer_clustering_from_rules.ipynb`

## Tham Khảo

- Agrawal, R., & Srikant, R. (1994). "Fast Algorithms for Mining Association Rules"
- Han, J., Pei, J., & Yin, Y. (2000). "Mining Frequent Patterns without Candidate Generation"
- Tan, P. N., Steinbach, M., & Kumar, V. (2005). "Introduction to Data Mining"
