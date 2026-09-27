# Báo cáo thí nghiệm Matrix Factorization trên MovieLens 100K

## 1. Tóm tắt

Project xây dựng một mô hình Matrix Factorization (MF) bằng PyTorch để dự
đoán rating phim. Mục tiêu chính là tạo baseline có thể so sánh với các mô
hình recommender phức tạp hơn, đồng thời đo ảnh hưởng của cách chia dữ liệu
đến bài toán cold-start.

Kết quả cho thấy MF đạt test RMSE `0.9638` trên random split với coverage
`99,78%`. Tuy nhiên, khi chia theo thời gian, test RMSE trên nhóm có thể đánh
giá là `1.0233` nhưng coverage chỉ `13,44%`; `85,82%` test interactions thuộc
user chưa xuất hiện trong train. Vì vậy, accuracy phải được trình bày cùng
coverage, nếu không kết luận sẽ quá lạc quan.

## 2. Dữ liệu

Project sử dụng MovieLens 100K (`data/ml-100k/u.data`):

| Thành phần   |                             Giá trị |
| ------------ | ----------------------------------: |
| Interactions |                             100.000 |
| Users        |                                 943 |
| Movies/items |                               1.682 |
| Rating       |                             1 đến 5 |
| Cột dữ liệu  | user_id, item_id, rating, timestamp |

Script kiểm tra dữ liệu trước khi chạy: file phải tồn tại, không có giá trị
thiếu, rating phải nằm trong `[1, 5]`, và không được có dòng trùng.

## 3. Mô hình và quy trình

Với user $u$ và item $i$, mô hình dự đoán:

$$
\hat r_{ui} = b_0 + b_u + b_i + p_u^T q_i
$$

Trong đó $b_0$ là global bias, $b_u$ và $b_i$ là user/item bias, còn $p_u$
và $q_i$ là embedding kích thước 64. Mô hình được tối ưu bằng Adam với:

- Batch size: 256
- Learning rate: 0,001
- Epochs: 20
- Seed: 42
- Loss: Mean Squared Error

Hai chiến lược chia dữ liệu được thử nghiệm:

1. **Random split:** xáo trộn interaction rồi chia 80% train, 10% validation,
   10% test.
2. **Temporal split:** sắp xếp theo timestamp rồi chia 80% train, 10%
   validation, 10% test.

Metric RMSE/MAE chỉ được tính cho dòng có cả user và item đã xuất hiện trong
train. Các dòng còn lại được đưa vào nhóm coverage riêng:

- `known_user_item`
- `unseen_user`
- `unseen_item`
- `unseen_user_item`

Đây là cách đánh giá phù hợp vì embedding của user/item chưa từng xuất hiện
trong train không phải là embedding đã được học.

## 4. Kết quả tổng quát

Kết quả lấy từ `results/mf_cold_start_final/summary.csv`.

| Split    | Train RMSE | Validation RMSE | Test RMSE | Test MAE | Test evaluable | Test coverage |
| -------- | ---------: | --------------: | --------: | -------: | -------------: | ------------: |
| Random   |     0,5124 |          0,9594 |    0,9638 |   0,7571 |   9.978/10.000 |        99,78% |
| Temporal |     0,5316 |          1,0432 |    1,0233 |   0,8179 |   1.344/10.000 |        13,44% |

### Nhận xét

- Random split có coverage gần như đầy đủ vì mỗi user thường xuất hiện ở cả
  train và test. Kết quả RMSE khoảng 0,964 thể hiện khả năng nội suy rating
  của MF trên dữ liệu MovieLens.
- Temporal split phản ánh tình huống triển khai thực tế hơn. Model chỉ đánh
  giá được 1.344 trên 10.000 test interactions; do đó test RMSE `1.0233` chỉ
  mô tả nhóm known user/item, không đại diện cho toàn bộ test set.
- Không nên kết luận temporal tốt hơn random chỉ vì RMSE temporal thấp hơn ở
  một lần chạy. Hai metric được tính trên hai tập evaluable rất khác nhau.

## 5. Phân tích cold-start

### 5.1. User/item chưa biết

Theo `evaluation_coverage.csv`:

| Split               | Known user/item | Unseen user | Unseen item | Unseen cả hai |
| ------------------- | --------------: | ----------: | ----------: | ------------: |
| Random validation   |           9.986 |           0 |          14 |             0 |
| Random test         |           9.978 |           0 |          22 |             0 |
| Temporal validation |           1.519 |       8.428 |          15 |            38 |
| Temporal test       |           1.344 |       8.507 |          74 |            75 |

Temporal test có 8.507 interactions từ 96 user chưa xuất hiện trong train.
MF thuần túy không thể học sở thích của các user này. Đây là giới hạn chính
của mô hình và cũng là lý do cần bổ sung side information hoặc cold-start
strategy.

### 5.2. Số interaction của user trong train

Ở temporal test:

- User chưa có interaction trong train: 8.582 dòng, 97 user; không thể tính
  RMSE/MAE trực tiếp bằng MF.
- User chỉ có 1 interaction trong train: 36 dòng, RMSE `1.5800`, MAE `1.2820`.
- User có từ 10 interaction trở lên: 1.382 dòng, RMSE `1.1420`, MAE `0.8862`.

Sai số cao hơn ở nhóm ít interaction cho thấy MF cần đủ lịch sử để học
embedding user ổn định. Đây là bằng chứng thực nghiệm cho ảnh hưởng của
cold-start, không chỉ là một giả định lý thuyết.

Ở random split, toàn bộ test user đều có ít nhất 10 interaction trong train
theo các nhóm đang báo cáo, nên các nhóm `k < 10` không có mẫu. Đây là hệ quả
của random split trên dataset mà mỗi user có tối thiểu 20 rating, không phải
lỗi thiếu dữ liệu của script.

## 6. Quá trình huấn luyện và overfitting

Epoch có validation RMSE tốt nhất:

| Split    | Epoch tốt nhất | Validation RMSE tốt nhất | Validation MAE tốt nhất |
| -------- | -------------: | -----------------------: | ----------------------: |
| Random   |             12 |                   0,9172 |                  0,7193 |
| Temporal |              9 |                   0,9785 |                  0,7580 |

Sau epoch tốt nhất, training loss tiếp tục giảm nhưng validation RMSE tăng.
Điều này cho thấy có dấu hiệu overfitting nhẹ, đặc biệt từ khoảng epoch 12
trên random split và epoch 9 trên temporal split. Phiên bản hiện tại chưa tự
động early stopping; đây là cải tiến nên thực hiện ở bước tiếp theo.

## 7. Kết luận

Matrix Factorization là baseline đơn giản nhưng hiệu quả cho known user/item.
Trên random split, model đạt sai số thấp và coverage cao. Tuy nhiên, khi dữ
liệu có tính thời gian, vấn đề cold-start trở nên rõ rệt: phần lớn test
interactions đến từ user chưa từng xuất hiện trong train.

Kết luận phù hợp để trình bày với giảng viên là:

> MF có khả năng dự đoán tốt trong bài toán nội suy, nhưng không giải quyết
> được user/item mới nếu chỉ dùng lịch sử rating. Vì vậy, một mô hình cải tiến
> cần báo cáo đồng thời accuracy và coverage, đồng thời bổ sung thông tin phụ
> hoặc chiến lược riêng cho cold-start.

## 8. Hạn chế

- Chỉ thử một seed và một cấu hình hyperparameter.
- Chưa có baseline đơn giản như global mean, user mean và item mean để so sánh.
- Chưa đánh giá recommendation top-K bằng Recall@K, Precision@K hoặc NDCG@K.
- Chưa dùng genre, thông tin nhân khẩu học, movie metadata hoặc knowledge graph.
- Validation/test known-group metrics không thể đánh giá trực tiếp unseen user.
- Kết quả được lấy trên một lần random split và một lần temporal split.

## 9. Hướng phát triển

1. Thêm baseline global mean, user mean và item mean.
2. Dùng early stopping theo validation RMSE và lưu best checkpoint.
3. Chạy nhiều seed hoặc 5-fold để báo cáo mean và standard deviation.
4. Xây dựng fallback cho unseen user/item bằng popularity hoặc bias model.
5. Kết hợp movie genre, user demographics và metadata để xử lý cold-start.
6. Bổ sung đánh giá top-K nếu mục tiêu cuối cùng là đề xuất phim.

## 10. Cách tái lập kết quả

Chạy từ thư mục project:

```bash
python mf_cold_start_experiment.py \
  --data-dir data \
  --output-dir results/mf_cold_start_final \
  --epochs 20 \
  --batch-size 256 \
  --dim 64 \
  --lr 0.001 \
  --seed 42
```

Các bảng kết quả nằm trong `results/mf_cold_start_final/` và sử dụng dấu
phân cách `|`. Hai file quan trọng nhất là `summary.csv` và
`evaluation_coverage.csv`; hai biểu đồ là `learning_curve_train_loss.png` và
`learning_curve_validation_rmse.png`.
