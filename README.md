# Matrix Factorization trên MovieLens 100K

## 1. Mục tiêu

Project xây dựng một baseline recommender system bằng Matrix Factorization
(MF) để dự đoán rating phim. Thí nghiệm tập trung vào hai câu hỏi:

1. MF hoạt động thế nào khi dữ liệu được chia ngẫu nhiên và theo thời gian?
2. Độ chính xác thay đổi thế nào đối với user có ít interaction trong train?

Đây là baseline để so sánh với các mô hình phức tạp hơn như KGAT hoặc mô hình
có thêm thông tin nội dung.

Báo cáo version 1 nằm trong [REPORT.md](REPORT.md); báo cáo cải tiến version 2
nằm trong [REPORT_v2.md](REPORT_v2.md).

## 2. Phương pháp

Với mỗi cặp user-item, model dự đoán:

```text
rating = global_bias + user_bias + item_bias + user_embedding . item_embedding
```

Model được tối ưu bằng Adam và MSE loss trong PyTorch.

Hai cách chia dữ liệu:

- `random`: interaction được xáo trộn rồi chia 80% train, 10% validation,
  10% test.
- `temporal`: interaction được sắp xếp theo timestamp rồi chia theo thứ tự
  thời gian 80%/10%/10%.

Seed được cố định để kết quả có thể tái lập.

## 3. Cài đặt và chạy

```bash
pip install numpy pandas matplotlib torch
python mf_cold_start_experiment.py
```

Có thể thay đổi tham số:

```bash
python mf_cold_start_experiment.py \
  --epochs 20 \
  --batch-size 256 \
  --dim 64 \
  --lr 0.001 \
  --seed 42 \
  --output-dir results/mf_cold_start_v2 \
  --patience 3 \
  --seeds 42,43,44
```

Script mặc định đọc `data/ml-100k/u.data` và tạo thư mục kết quả nếu chưa có.
Các file dữ liệu kết quả dạng bảng được lưu với dấu phân cách `|` thay vì dấu
phẩy để dễ đọc khi mở trong editor hoặc import vào spreadsheet.

## 4. Cách đọc kết quả

Các file chính trong `results/mf_cold_start_v2/`:

- `summary.csv`: RMSE/MAE tổng quát cho train, validation và test.
- `split_statistics.csv`: số user/item trong mỗi split và số đối tượng unseen.
- `evaluation_coverage.csv`: phân loại dòng thành `known_user_item`,
  `unseen_user`, `unseen_item`, `unseen_user_item`.
- `cold_start_groups_*.csv`: kết quả theo số interaction của user trong train.
- `cold_start_k_*.csv`: kết quả với quy ước cold-start `train_interactions < k`.
- `training_*.csv`: loss và validation metrics theo epoch.
- `learning_curve_*.png`: biểu đồ quá trình huấn luyện.
- `baseline_results.csv`: kết quả từng seed của global/user/item mean,
  bias fallback và MF + fallback.
- `baseline_mean_std.csv`: mean/std của các model.
- `summary_mean_std.csv`: mean/std của MF trên nhiều seed.
- `fallback_coverage.csv`: metric fallback theo nhóm known/unseen.
- `mf_*_best.pt`: best checkpoint theo split và seed.

RMSE/MAE chỉ được tính cho nhóm `known_user_item`, vì embedding của user và
item unseen chưa được học từ train. Các nhóm unseen vẫn được đếm trong
`evaluation_coverage.csv` để báo cáo rõ giới hạn coverage thay vì đưa dự đoán
ngẫu nhiên vào metric chính.

## 5. Kết quả hiện có

Kết quả version 2 với ba seed cho thấy:

- MF random test RMSE trung bình `0.9201 ± 0.0077`.
- MF temporal test RMSE trung bình `0.9997 ± 0.0134` trên nhóm known
  user/item.
- MF + fallback đánh giá được toàn bộ test, đạt RMSE trung bình `1.0361` trên
  temporal split.
- Item mean là baseline đơn giản mạnh nhất, với RMSE `1.0264` random và
  `1.0469` temporal.

Version 1 vẫn được lưu trong `version1/`. Dùng [REPORT_v2.md](REPORT_v2.md) để
trình bày kết quả cải tiến và `fallback_coverage.csv` để giải thích các nhóm
unseen.

## 6. Giới hạn và hướng phát triển

- MF chỉ dùng lịch sử rating, chưa dùng genre, thông tin user hoặc knowledge
  graph.
- User/item unseen cần cold-start strategy riêng, chẳng hạn popularity prior,
  content features hoặc side-information encoder.
- Có thể mở rộng bằng nhiều seed, 5-fold evaluation, early stopping và khoảng
  tin cậy cho RMSE/MAE.
- Khi chuyển từ rating prediction sang top-K recommendation, cần bổ sung
  Recall@K, Precision@K hoặc NDCG@K.

## 7. Cách trình bày với giảng viên

Thông điệp chính nên là: **baseline không chỉ báo cáo độ chính xác mà còn báo
cáo model có thể đánh giá bao phủ bao nhiêu dữ liệu**. Random split đo năng
lực nội suy; temporal split phản ánh gần hơn tình huống user/item mới. Vì vậy
coverage và phân nhóm unseen là phần bắt buộc để kết luận không bị lạc quan
giả tạo.
