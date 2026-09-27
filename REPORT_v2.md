# Báo cáo version 2: Matrix Factorization và xử lý cold-start

## 1. Phạm vi cải tiến

Version 2 kế thừa kết quả version 1 và thực hiện bốn cải tiến:

1. Bổ sung ba baseline: global mean, user mean và item mean.
2. Thêm early stopping theo validation RMSE và lưu best checkpoint.
3. Chạy ba seed `42, 43, 44`, báo cáo mean và standard deviation.
4. Thêm bias fallback cho user/item unseen để dự đoán được toàn bộ test set.

Kết quả version 1 được bảo toàn tại `version1/`, gồm báo cáo cũ và thư mục
`version1/results_mf_cold_start_final/`.

## 2. Thiết kế version 2

### 2.1. Baseline

- **Global mean:** mọi dự đoán bằng rating trung bình của train.
- **User mean:** dùng rating trung bình của user; user unseen fallback về
  global mean.
- **Item mean:** dùng rating trung bình của item; item unseen fallback về
  global mean.

Các baseline chỉ được fit trên train, không dùng validation/test để tránh
data leakage.

### 2.2. Bias fallback

Fallback phân cấp được định nghĩa như sau:

| Trạng thái                | Dự đoán                 |
| ------------------------- | ----------------------- |
| User và item đã biết      | user mean ưu tiên trước |
| User unseen, item đã biết | item mean               |
| User đã biết, item unseen | user mean               |
| User và item đều unseen   | global mean             |

Với `mf_with_bias_fallback`, MF được dùng cho known user/item; các trường hợp
unseen được chuyển sang fallback tương ứng. Nhờ đó model có metric trên toàn
bộ test thay vì bỏ qua phần cold-start.

### 2.3. Early stopping và checkpoint

Mỗi split/seed lưu một checkpoint:

```text
mf_random_seed42_best.pt
mf_temporal_seed42_best.pt
```

Checkpoint được lưu khi validation RMSE tốt hơn lần trước. Nếu RMSE không cải
thiện trong `patience=3` epoch liên tiếp, huấn luyện dừng; sau đó best
checkpoint được nạp lại trước khi đánh giá test.

### 2.4. Multi-seed

Mỗi seed tạo random split riêng. Temporal split không đổi vì được xác định
bởi timestamp. Các bảng mean/std được tính trên ba lần chạy độc lập.

## 3. Cấu hình và dữ liệu

- Dataset: MovieLens 100K, 100.000 interactions, 943 users, 1.682 items.
- Split: 80% train, 10% validation, 10% test.
- Embedding dimension: 64.
- Batch size: 256.
- Learning rate: 0,001.
- Maximum epochs: 20.
- Early stopping patience: 3.
- Seeds: 42, 43, 44.
- Loss: MSE; metrics: RMSE và MAE.

## 4. Kết quả MF với multi-seed

Nguồn: `results/mf_cold_start_v2/summary_mean_std.csv`.

| Split    | Test RMSE mean ± std | Test MAE mean ± std | Epoch tốt nhất trung bình |
| -------- | -------------------: | ------------------: | ------------------------: |
| Random   |      0,9201 ± 0,0077 |     0,7277 ± 0,0075 |                     11,67 |
| Temporal |      0,9997 ± 0,0134 |     0,8016 ± 0,0125 |                      8,67 |

MF version 2 tốt hơn kết quả version 1 cố định 20 epoch ở random split vì
checkpoint được lấy tại epoch validation tốt nhất thay vì epoch cuối. Temporal
có epoch tốt nhất thấp hơn, phù hợp với dấu hiệu overfitting sớm.

## 5. So sánh baseline

Nguồn: `results/mf_cold_start_v2/baseline_mean_std.csv`. Các baseline trong
bảng được đánh giá trên toàn bộ 10.000 test interactions.

| Split    | Model         | RMSE mean ± std |  MAE mean ± std |
| -------- | ------------- | --------------: | --------------: |
| Random   | Global mean   | 1,1313 ± 0,0043 | 0,9501 ± 0,0035 |
| Random   | User mean     | 1,0469 ± 0,0058 | 0,8400 ± 0,0045 |
| Random   | Item mean     | 1,0264 ± 0,0055 | 0,8203 ± 0,0059 |
| Random   | MF + fallback | 0,9201 ± 0,0076 | 0,7277 ± 0,0074 |
| Temporal | Global mean   | 1,1399 ± 0,0000 | 0,9694 ± 0,0000 |
| Temporal | User mean     | 1,1512 ± 0,0000 | 0,9717 ± 0,0000 |
| Temporal | Item mean     | 1,0469 ± 0,0000 | 0,8476 ± 0,0000 |
| Temporal | MF + fallback | 1,0361 ± 0,0017 | 0,8345 ± 0,0017 |

Item mean là baseline đơn giản mạnh nhất trong hai split. MF + fallback vẫn
tốt hơn các baseline trên random và temporal khi metric được tính trên toàn bộ
test. Đây là so sánh công bằng hơn so với chỉ tính MF trên known user/item.

## 6. Coverage và cold-start

Temporal test có cấu trúc:

| Nhóm            | Số interactions |  Tỷ lệ |
| --------------- | --------------: | -----: |
| Known user/item |           1.344 | 13,44% |
| Unseen user     |           8.507 | 85,07% |
| Unseen item     |              74 |  0,74% |
| Unseen cả hai   |              75 |  0,75% |

Nếu chỉ dùng MF, chỉ 1.344 dòng được đánh giá trực tiếp. Với fallback, toàn bộ
10.000 dòng đều có dự đoán:

| Temporal test group | Bias fallback RMSE | MF + fallback RMSE |
| ------------------- | -----------------: | -----------------: |
| Known user/item     |              1,275 |      1,000 ± 0,013 |
| Unseen user         |              1,040 |              1,040 |
| Unseen item         |              1,086 |              1,086 |
| Unseen cả hai       |              1,198 |              1,198 |

Ở nhóm unseen, MF không thể tạo embedding đã học nên fallback được dùng. Ở
nhóm known, MF tiếp tục phát huy lợi thế latent factors.

## 7. Ý nghĩa kết quả

1. **Accuracy:** MF có latent factors giúp giảm RMSE rõ rệt so với global,
   user và item mean trên random split.
2. **Temporal realism:** temporal split khó hơn vì nhiều user mới xuất hiện
   sau mốc train; đây là tình huống gần với triển khai thực tế.
3. **Cold-start coverage:** fallback không làm user mới trở thành user đã học,
   nhưng cung cấp dự đoán có nguyên tắc thay vì bỏ trống.
4. **Ổn định:** std nhỏ qua ba seed cho thấy kết quả không phụ thuộc mạnh vào
   một lần khởi tạo hoặc một random split.
5. **Regularization by stopping:** epoch tốt nhất trung bình là 11,67 cho
   random và 8,67 cho temporal; chạy đến epoch tối đa không phải lựa chọn tối
   ưu.

## 8. Hạn chế

- Bias fallback vẫn chưa dùng genre, demographics hoặc metadata của phim.
- User/item hoàn toàn mới chỉ nhận global mean, chưa có content encoder.
- Mới chạy ba seed, chưa phải 5-fold cross-validation.
- Chưa đánh giá top-K recommendation bằng Recall@K, Precision@K hoặc NDCG@K.
- Fallback dùng mean rating, chưa tối ưu riêng cho ranking hoặc calibration.

## 9. Kết luận trình bày với giảng viên

> Version 2 bổ sung baseline để chứng minh MF thực sự tạo giá trị, dùng early
> stopping để tránh chọn model theo epoch cuối, dùng nhiều seed để đo độ ổn
> định, và thêm fallback để xử lý toàn bộ user/item unseen. Kết quả cho thấy
> MF tốt nhất ở nhóm known user/item, còn fallback là thành phần bắt buộc để
> hệ thống có thể phục vụ dữ liệu cold-start trong thực tế.

## 10. Tái lập kết quả

```bash
python mf_cold_start_experiment.py \
  --data-dir data \
  --output-dir results/mf_cold_start_v2 \
  --epochs 20 \
  --patience 3 \
  --batch-size 256 \
  --dim 64 \
  --lr 0.001 \
  --seeds 42,43,44
```

Các file quan trọng:

- `summary_mean_std.csv`: mean/std của MF.
- `baseline_mean_std.csv`: mean/std của các baseline và MF + fallback.
- `baseline_results.csv`: kết quả từng seed trên toàn bộ test.
- `fallback_coverage.csv`: metric fallback theo nhóm known/unseen.
- `mf_*_best.pt`: best checkpoint của từng split/seed.
- `training_*_seed*.csv`: lịch sử train và validation từng seed.
