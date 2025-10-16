# Fake-News-Detector

Dự án cung cấp bộ công cụ thu thập dữ liệu báo chí tài chính - kinh tế Việt Nam và huấn luyện mô hình phát hiện tin giả.

## Tính năng chính

- Thu thập bài viết mới nhất từ nhiều nguồn báo tài chính Việt Nam (VnExpress, CafeF, Vietstock, v.v.) thông qua RSS.
- Làm sạch văn bản tiếng Việt, loại bỏ dấu câu và stopwords phổ biến.
- Huấn luyện mô hình Logistic Regression với TF-IDF để phân loại tin thật/tin giả.
- Công cụ dòng lệnh để thu thập dữ liệu, huấn luyện và dự đoán.
- Hỗ trợ ghi nhãn thủ công nhằm mở rộng tập dữ liệu.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Hoặc cài đặt qua `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Sử dụng

### Thu thập dữ liệu

```bash
python -m fake_news_detector.cli collect --limit 30 --output data/articles.json
```

### Huấn luyện mô hình

```bash
python -m fake_news_detector.cli train data/sample_labeled_news.csv content label \
    --model-output models/fake_news_detector.joblib
```

### Dự đoán

```bash
python -m fake_news_detector.cli predict models/fake_news_detector.joblib data/sample_labeled_news.csv \
    --output data/predictions.csv
```

## Cấu trúc thư mục

```
.
├── data
│   └── sample_labeled_news.csv
├── pyproject.toml
├── requirements.txt
└── src
    └── fake_news_detector
        ├── __init__.py
        ├── cli.py
        ├── data_collection.py
        ├── model.py
        ├── pipeline.py
        └── preprocessing.py
```

## Ghi chú

- Việc thu thập dữ liệu cần kết nối Internet và có thể bị chặn bởi một số trang báo.
- Để cải thiện độ chính xác, hãy mở rộng tập dữ liệu gán nhãn và thử nghiệm các mô hình khác như XGBoost, transformer tiếng Việt (PhoBERT).
- Khi triển khai thực tế, cân nhắc xây dựng lịch trình thu thập dữ liệu định kỳ và hệ thống kiểm duyệt thủ công để đảm bảo chất lượng.
