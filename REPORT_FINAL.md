# Báo cáo cuối cùng: Matrix Factorization và đánh giá Cold-start trên MovieLens 100K

## 1. Tóm tắt

Báo cáo này tổng hợp toàn bộ quá trình xây dựng và đánh giá mô hình Matrix Factorization (MF) trên bộ dữ liệu MovieLens 100K. Mục tiêu là tạo một baseline recommender system có thể:

- dự đoán rating của user đối với item;
- so sánh khả năng nội suy giữa random split và temporal split;
- đo ảnh hưởng của số lượng interaction trong train đến chất lượng dự đoán;
- nhận diện giới hạn của MF khi gặp user hoặc item chưa xuất hiện trong train;
- bổ sung fallback để tạo dự đoán cho toàn bộ test set.

Kết quả chính:

- Random split: MF đạt test RMSE trung bình `0.9201 ± 0.0077` trên nhóm known user/item, với coverage khoảng `99.79%`.
- Temporal split: MF đạt test RMSE trung bình `0.9997 ± 0.0134` trên nhóm known user/item, nhưng coverage chỉ `13.44%`.
- Khi dùng bias fallback, toàn bộ test set đều có dự đoán. MF + fallback đạt RMSE trung bình `0.9201` trên random và `1.0361` trên temporal.
- Temporal split khó hơn rõ rệt vì phần lớn test interactions đến từ user chưa xuất hiện trong train.
- Early stopping giúp chọn checkpoint tốt nhất thay vì sử dụng mù quáng epoch cuối.

## 2. Mục tiêu và câu hỏi nghiên cứu

Thí nghiệm được thiết kế để trả lời bốn câu hỏi:

1. MF dự đoán rating tốt đến mức nào trên MovieLens 100K?
2. Kết quả thay đổi ra sao giữa random split và temporal split?
3. Khi user có ít hoặc không có interaction trong train, sai số và khả năng đánh giá thay đổi thế nào?
4. Các baseline đơn giản và chiến lược fallback có thể xử lý phần dữ liệu unseen đến đâu?

Hai loại split có ý nghĩa khác nhau:

- **Random split** chủ yếu đo khả năng nội suy khi các user và item thường xuất hiện ở cả train lẫn test.
- **Temporal split** gần với tình huống triển khai thực tế hơn, trong đó dữ liệu tương lai có thể chứa user hoặc item chưa từng xuất hiện trong giai đoạn huấn luyện.

Vì vậy, chỉ báo cáo RMSE mà không báo cáo coverage sẽ dẫn đến kết luận quá lạc quan, đặc biệt với temporal split.

## 3. Dữ liệu

### 3.1. Bộ dữ liệu

Thí nghiệm sử dụng MovieLens 100K, file `data/ml-100k/u.data`.

| Thuộc tính      |                                     Giá trị |
| --------------- | ------------------------------------------: |
| Số interactions |                                     100.000 |
| Số user         |                                         943 |
| Số item/phim    |                                       1.682 |
| Khoảng rating   |                                     1 đến 5 |
| Các cột         | `user_id`, `item_id`, `rating`, `timestamp` |

Script kiểm tra dữ liệu trước khi chạy:

- file dữ liệu phải tồn tại;
- không được có giá trị thiếu;
- rating phải nằm trong đoạn `[1, 5]`;
- không được có interaction trùng hoàn toàn.

### 3.2. Chia dữ liệu

Mỗi thí nghiệm dùng tỷ lệ:

- 80% train: 80.000 interactions;
- 10% validation: 10.000 interactions;
- 10% test: 10.000 interactions.

#### Random split

Toàn bộ interactions được xáo trộn bằng `random_state=seed`, sau đó cắt theo tỷ lệ 80/10/10. Mỗi seed tạo ra một random split riêng.

#### Temporal split

Interactions được sắp xếp tăng dần theo `timestamp` bằng stable sort, sau đó chia theo thứ tự thời gian. Với cùng dataset, temporal split không thay đổi theo seed.

Cách chia này mô phỏng việc huấn luyện trên lịch sử và đánh giá trên các tương tác xảy ra sau đó.

## 4. Mô hình Matrix Factorization

Mô hình trong [mf_cold_start_experiment.py](mf_cold_start_experiment.py) dùng embedding cho user và item, kết hợp với bias:

$$
\hat{r}_{ui} = b_0 + b_u + b_i + p_u^T q_i
$$

Trong đó:

- $\hat{r}_{ui}$ là rating dự đoán cho user $u$ và item $i$;
- $b_0$ là global bias;
- $b_u$ là user bias;
- $b_i$ là item bias;
- $p_u$ là user embedding;
- $q_i$ là item embedding.

Kích thước embedding là 64. Embedding được khởi tạo theo phân phối chuẩn với độ lệch chuẩn `0.05`; các bias được khởi tạo bằng 0.

Mô hình được tối ưu bằng:

- loss: Mean Squared Error (MSE);
- optimizer: Adam;
- learning rate: `0.001`;
- batch size: `256`;
- tối đa: `20` epochs;
- early stopping patience: `3` epochs;
- seeds: `42`, `43`, `44`.

## 5. Quy trình huấn luyện và đánh giá

### 5.1. Encoding ID

User ID và item ID được ánh xạ thành index liên tục dựa trên các ID xuất hiện trong toàn bộ dataset của split. Việc này chỉ dùng thông tin định danh để embedding lookup không bị lỗi; model vẫn chỉ được cập nhật bằng interactions trong train.

### 5.2. Validation

MF chỉ được đánh giá bằng RMSE/MAE trên các dòng validation có cả user và item đã xuất hiện trong train. Các dòng chứa user hoặc item unseen không được đưa vào validation metric của MF vì embedding tương ứng chưa được học.

Checkpoint được lưu khi validation RMSE tốt hơn giá trị tốt nhất trước đó. Sau khi huấn luyện kết thúc, checkpoint tốt nhất được nạp lại trước khi đánh giá test.

### 5.3. Metric

Với prediction $\hat{y}_j$ và rating thật $y_j$:

$$
RMSE = \sqrt{\frac{1}{n}\sum_{j=1}^{n}(\hat{y}_j-y_j)^2}
$$

$$
MAE = \frac{1}{n}\sum_{j=1}^{n}|\hat{y}_j-y_j|
$$

RMSE phạt mạnh các sai số lớn; MAE dễ diễn giải hơn theo đơn vị rating.

### 5.4. Coverage

Mỗi dòng validation/test được phân loại thành bốn nhóm:

- `known_user_item`: user và item đều xuất hiện trong train;
- `unseen_user`: user chưa xuất hiện trong train, item đã biết;
- `unseen_item`: user đã biết, item chưa xuất hiện trong train;
- `unseen_user_item`: cả user và item đều chưa xuất hiện trong train.

MF thuần túy chỉ đánh giá trực tiếp được nhóm `known_user_item`.

## 6. Baseline và bias fallback

### 6.1. Baseline

Ba baseline được fit chỉ trên train:

1. **Global mean:** dự đoán bằng rating trung bình toàn train.
2. **User mean:** dự đoán bằng rating trung bình của user; user unseen dùng global mean.
3. **Item mean:** dự đoán bằng rating trung bình của item; item unseen dùng global mean.

Các baseline này giúp xác định MF có thực sự tạo thêm giá trị so với các dự đoán trung bình đơn giản hay không.

### 6.2. Bias fallback

Fallback phân cấp dùng thứ tự:

| Trạng thái                | Prediction  |
| ------------------------- | ----------- |
| User và item đã biết      | user mean   |
| User unseen, item đã biết | item mean   |
| User đã biết, item unseen | user mean   |
| User và item đều unseen   | global mean |

Model `mf_with_bias_fallback` dùng MF cho nhóm `known_user_item`. Những nhóm còn lại dùng fallback tương ứng. Nhờ đó, toàn bộ test set đều nhận được prediction.

Fallback không làm MF học được user mới. Nó chỉ cung cấp một prediction có nguyên tắc cho các trường hợp mà embedding MF không thể sử dụng.

## 7. Kết quả huấn luyện MF

### 7.1. Kết quả theo từng seed

Các số liệu lấy từ `results/mf_cold_start_v3/summary.csv`.

| Split    | Seed | Train RMSE | Val RMSE | Best epoch | Test RMSE known | Test MAE known | Test evaluable |
| -------- | ---: | ---------: | -------: | ---------: | --------------: | -------------: | -------------: |
| Random   |   42 |     0.7138 |   0.9172 |         12 |          0.9203 |         0.7250 |   9.978/10.000 |
| Temporal |   42 |     0.7929 |   0.9785 |          9 |          1.0149 |         0.8156 |   1.344/10.000 |
| Random   |   43 |     0.7409 |   0.9182 |         11 |          0.9277 |         0.7361 |   9.978/10.000 |
| Temporal |   43 |     0.7894 |   0.9904 |          9 |          0.9894 |         0.7918 |   1.344/10.000 |
| Random   |   44 |     0.7193 |   0.9185 |         12 |          0.9123 |         0.7218 |   9.982/10.000 |
| Temporal |   44 |     0.8130 |   0.9814 |          8 |          0.9948 |         0.7974 |   1.344/10.000 |

### 7.2. Trung bình và độ lệch chuẩn

| Split    | Test RMSE mean ± std | Test MAE mean ± std | Best epoch mean |
| -------- | -------------------: | ------------------: | --------------: |
| Random   |    `0.9201 ± 0.0077` |   `0.7277 ± 0.0075` |           11.67 |
| Temporal |    `0.9997 ± 0.0134` |   `0.8016 ± 0.0125` |            8.67 |

Random split có sai số thấp hơn temporal split. Tuy nhiên, cần lưu ý hai RMSE này được tính trên tập evaluable khác nhau rất nhiều: random gần như toàn bộ test, còn temporal chỉ trên nhóm known user/item.

Độ lệch chuẩn nhỏ qua ba seed cho thấy kết quả tương đối ổn định trong cấu hình hiện tại.

## 8. Coverage và cold-start

### 8.1. Coverage trên validation/test

#### Random split

- Validation: 9.986/10.000 dòng evaluable, tương đương `99.86%`.
- Test: 9.978 đến 9.982/10.000 dòng evaluable, khoảng `99.78%` đến `99.82%`.
- Phần không evaluable chủ yếu là unseen item; không có unseen user.

#### Temporal split

- Validation: 1.519/10.000 dòng evaluable, tương đương `15.19%`.
- Test: 1.344/10.000 dòng evaluable, tương đương `13.44%`.
- Test có 8.507 interactions thuộc nhóm unseen user, 74 thuộc unseen item và 75 thuộc unseen user/item.

Bảng temporal test tổng hợp:

| Nhóm             | Interactions |  Tỷ lệ | MF trực tiếp |
| ---------------- | -----------: | -----: | ------------ |
| Known user/item  |        1.344 | 13.44% | Có           |
| Unseen user      |        8.507 | 85.07% | Không        |
| Unseen item      |           74 |  0.74% | Không        |
| Unseen user/item |           75 |  0.75% | Không        |
| Tổng             |       10.000 |   100% | -            |

Kết quả này cho thấy khó khăn chính của temporal split là user cold-start, không phải item cold-start.

### 8.2. Phân tích theo số interaction trong train

Bảng `cold_start_k_all.csv` dùng các ngưỡng:

- `k=10`: user có ít hơn 10 train interactions;
- `k=5`: user có ít hơn 5 train interactions;
- `k=2`: user có ít hơn 2 train interactions;
- `k=1`: user có 0 train interactions.

Với temporal test:

| Điều kiện               | Interactions | Users | RMSE known users | MAE known users |
| ----------------------- | -----------: | ----: | ---------------: | --------------: |
| train interactions < 10 |        8.618 |    98 |           1.8492 |          1.5705 |
| train interactions < 5  |        8.618 |    98 |           1.8492 |          1.5705 |
| train interactions < 2  |        8.618 |    98 |           1.8492 |          1.5705 |
| train interactions < 1  |        8.582 |    97 |         Không có |        Không có |

Ba ngưỡng đầu có cùng số interactions vì nhóm này chủ yếu là user chưa từng xuất hiện trong train. Khi loại các user có 0 interaction, nhóm user có đúng 1 interaction trong train có 36 test interactions; MF đạt RMSE khoảng `1.7564` đến `1.8492` tùy seed.

## 9. Kết quả MF riêng

Đây là kết quả của **MF thuần túy**, không dùng fallback. Model chỉ dự đoán
được các dòng mà cả user và item đã xuất hiện trong train. Vì vậy, metric ở
bảng này chỉ tính trên nhóm `known_user_item`, không phải toàn bộ 10.000 test
interactions.

| Split    | Test RMSE mean ± std | Test MAE mean ± std |   Số dòng evaluable |      Coverage |
| -------- | -------------------: | ------------------: | ------------------: | ------------: |
| Random   |    `0.9201 ± 0.0077` |   `0.7277 ± 0.0075` | khoảng 9.978/10.000 | khoảng 99.79% |
| Temporal |    `0.9997 ± 0.0134` |   `0.8016 ± 0.0125` |        1.344/10.000 |        13.44% |

Đây là metric phù hợp để trả lời câu hỏi: **MF học tốt đến đâu khi user và
item đã có lịch sử trong train?** Temporal RMSE không được hiểu là kết quả
cho toàn bộ temporal test, vì 8.656/10.000 dòng còn lại không thể được MF
thuần túy đánh giá trực tiếp.

## 10. So sánh với baseline trên toàn bộ test

Các số liệu lấy từ `baseline_mean_std.csv` và được tính trên toàn bộ 10.000 test interactions, bao gồm cả unseen user/item.

### 10.1. Random split

| Model              |   RMSE mean ± std |    MAE mean ± std |
| ------------------ | ----------------: | ----------------: |
| Global mean        | `1.1313 ± 0.0043` | `0.9501 ± 0.0035` |
| User mean          | `1.0469 ± 0.0058` | `0.8400 ± 0.0045` |
| Item mean          | `1.0264 ± 0.0055` | `0.8203 ± 0.0059` |
| Bias fallback      | `1.0469 ± 0.0058` | `0.8400 ± 0.0045` |
| MF + bias fallback | `0.9201 ± 0.0076` | `0.7277 ± 0.0074` |

MF + fallback tốt hơn item mean khoảng `0.1063` RMSE tuyệt đối. Đây là so
sánh trên toàn bộ test set, trong đó MF được dùng cho known user/item và
fallback được dùng cho các trường hợp unseen. Vì vậy, kết quả này không được
gọi là MF thuần túy.

### 10.2. Temporal split

| Model              |   RMSE mean ± std |    MAE mean ± std |
| ------------------ | ----------------: | ----------------: |
| Global mean        | `1.1399 ± 0.0000` | `0.9694 ± 0.0000` |
| User mean          | `1.1512 ± 0.0000` | `0.9717 ± 0.0000` |
| Item mean          | `1.0469 ± 0.0000` | `0.8476 ± 0.0000` |
| Bias fallback      | `1.0759 ± 0.0000` | `0.8698 ± 0.0000` |
| MF + bias fallback | `1.0361 ± 0.0017` | `0.8345 ± 0.0017` |

MF + fallback vẫn tốt hơn các baseline khi đánh giá toàn bộ temporal test,
nhưng mức cải thiện nhỏ hơn random split. Nguyên nhân là 85.07% temporal test
thuộc unseen user và phải dùng fallback, không thể hưởng lợi từ user embedding.

Tóm lại, hai loại kết quả cần được phân biệt:

| Cách đánh giá      | Phạm vi               | Ý nghĩa                                           |
| ------------------ | --------------------- | ------------------------------------------------- |
| MF riêng           | Chỉ `known_user_item` | Đo năng lực latent factors của MF                 |
| MF + bias fallback | Toàn bộ test set      | Đo khả năng phục vụ dữ liệu thực tế có cold-start |

## 11. Phân tích theo coverage group

Temporal test cho seed 42 minh họa rõ cách model hoạt động:

| Nhóm             | Số dòng | Bias fallback RMSE | MF + fallback RMSE |
| ---------------- | ------: | -----------------: | -----------------: |
| Known user/item  |   1.344 |             1.2747 |             1.0149 |
| Unseen user      |   8.507 |             1.0398 |             1.0398 |
| Unseen item      |      74 |             1.0860 |             1.0860 |
| Unseen user/item |      75 |             1.1978 |             1.1978 |

MF cải thiện rõ rệt ở nhóm known user/item: RMSE giảm từ `1.2747` xuống `1.0149`. Với các nhóm unseen, kết quả của MF + fallback giống bias fallback vì model không có embedding đã học cho đối tượng chưa xuất hiện trong train.

Đây là bằng chứng quan trọng cho hai kết luận:

1. MF có giá trị trong bài toán nội suy known user/item.
2. MF thuần túy chưa giải quyết được cold-start; cần metadata hoặc một content encoder nếu muốn dự đoán tốt hơn cho user/item mới.

## 12. Hạn chế

1. **Chưa sử dụng side information:** model chỉ dùng lịch sử rating, chưa dùng genre, occupation, age, gender, movie metadata hoặc knowledge graph.
2. **Cold-start user vẫn còn yếu:** user hoàn toàn mới chỉ được dự đoán bằng item mean hoặc global mean tùy trạng thái item.
3. **Cold-start item chưa được học semantic:** item mới không có embedding được cập nhật từ train.
4. **Chỉ có ba seed:** kết quả đã có độ ổn định sơ bộ nhưng chưa thay thế cho 5-fold cross-validation hoặc khoảng tin cậy chặt chẽ hơn.
5. **Chưa đánh giá top-K recommendation:** hiện tại chỉ đánh giá rating prediction bằng RMSE/MAE, chưa có Recall@K, Precision@K hoặc NDCG@K.
6. **Temporal split phụ thuộc dataset:** kết quả phản ánh đặc tính timestamp của MovieLens 100K và có thể thay đổi trên dataset khác.
7. **Fallback tối ưu cho rating prediction:** fallback hiện dùng mean rating, chưa được thiết kế riêng cho ranking recommendation.

## 13. Hướng phát triển

Các hướng cải tiến hợp lý tiếp theo:

1. Dùng genre và metadata phim để xây content embedding cho item mới.
2. Dùng thông tin nhân khẩu học user để xây user representation cho user mới.
3. Kết hợp MF với neural network hoặc factorization machine có side features.
4. Huấn luyện riêng một popularity/bias model cho nhóm cold-start.
5. Bổ sung ranking loss và đánh giá top-K.
6. Chạy thêm seeds hoặc cross-validation để tăng độ tin cậy thống kê.
7. Báo cáo thêm khoảng tin cậy, phân phối lỗi và hiệu năng theo từng nhóm rating.

## 14. Cách tái lập

Chạy từ thư mục `mf-with-movicelens100k`:

```bash
python mf_cold_start_experiment.py \
  --data-dir data \
  --output-dir results/mf_cold_start_v3 \
  --epochs 20 \
  --patience 3 \
  --batch-size 256 \
  --dim 64 \
  --lr 0.001 \
  --seeds 42,43,44
```

Các file quan trọng:

- `summary.csv`: metric MF theo từng split và seed;
- `summary_mean_std.csv`: trung bình và độ lệch chuẩn của MF;
- `baseline_results.csv`: kết quả từng seed của các baseline;
- `baseline_mean_std.csv`: tổng hợp baseline;
- `evaluation_coverage.csv`: nhóm known/unseen;
- `fallback_coverage.csv`: metric theo coverage group;
- `cold_start_groups_*.csv`: metric theo số interaction trong train;
- `cold_start_k_*.csv`: metric theo các ngưỡng cold-start;
- `training_*_seed*.csv`: loss và validation metric theo epoch;
- `mf_*_best.pt`: checkpoint tốt nhất của từng split và seed.

Các file CSV dùng dấu phân cách `|`.

## 15. Kết luận cuối cùng

Matrix Factorization là một baseline mạnh cho bài toán dự đoán rating khi user và item đã xuất hiện trong dữ liệu huấn luyện. Trên random split, model đạt RMSE khoảng `0.92` và gần như bao phủ toàn bộ test set. Khi chuyển sang temporal split, RMSE trên nhóm known vẫn ở mức khoảng `1.00`, nhưng coverage giảm xuống `13.44%` vì phần lớn test interactions thuộc user chưa từng xuất hiện trong train.

Bias fallback giúp hệ thống tạo prediction cho 100% test interactions, nhưng không thể thay thế việc học representation cho user/item mới. Kết quả tốt nhất đến từ sự kết hợp: dùng MF cho known user/item và fallback cho unseen cases.

Thông điệp chính của thực nghiệm là:

> MF học tốt quan hệ user-item trong bài toán nội suy, nhưng không tự giải quyết được cold-start. Vì vậy, một đánh giá đúng phải trình bày đồng thời accuracy, coverage và phân tích theo nhóm known/unseen.
