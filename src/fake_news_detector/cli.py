"""Command line interface for the fake news detection toolkit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import batch_predict, collect_and_save, train_from_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thu thập dữ liệu báo chí tài chính Việt Nam và phát hiện tin giả."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Thu thập bài viết mới nhất")
    collect_parser.add_argument(
        "--sources",
        nargs="*",
        help="Danh sách nguồn tin (mặc định: tất cả)",
    )
    collect_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Số lượng bài viết tối đa mỗi nguồn",
    )
    collect_parser.add_argument(
        "--enrich",
        action="store_true",
        help="Tải nội dung đầy đủ của bài viết",
    )
    collect_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/articles.json"),
        help="Đường dẫn file JSON để lưu kết quả",
    )

    train_parser = subparsers.add_parser("train", help="Huấn luyện mô hình phát hiện tin giả")
    train_parser.add_argument("dataset", type=Path, help="File CSV chứa dữ liệu đã gán nhãn")
    train_parser.add_argument("text_column", type=str, help="Cột văn bản")
    train_parser.add_argument("label_column", type=str, help="Cột nhãn (0/1)")
    train_parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/fake_news_detector.joblib"),
        help="Đường dẫn lưu mô hình",
    )
    train_parser.add_argument(
        "--config-output",
        type=Path,
        default=Path("models/config.json"),
        help="Đường dẫn lưu cấu hình",
    )

    predict_parser = subparsers.add_parser("predict", help="Dự đoán nhãn cho dữ liệu mới")
    predict_parser.add_argument("model", type=Path, help="File mô hình đã huấn luyện")
    predict_parser.add_argument("input", type=Path, help="File CSV cần dự đoán")
    predict_parser.add_argument(
        "--text-column",
        type=str,
        default="content",
        help="Tên cột văn bản trong file",
    )
    predict_parser.add_argument(
        "--output",
        type=Path,
        help="File CSV lưu kết quả dự đoán",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.command == "collect":
        frame = collect_and_save(
            output_path=args.output,
            sources=args.sources,
            limit_per_source=args.limit,
            enrich=args.enrich,
        )
        print(frame.head().to_string())
    elif args.command == "train":
        metrics = train_from_dataset(
            dataset_path=args.dataset,
            text_column=args.text_column,
            label_column=args.label_column,
            model_output=args.model_output,
            config_path=args.config_output,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    elif args.command == "predict":
        frame = batch_predict(
            model_path=args.model,
            input_path=args.input,
            text_column=args.text_column,
            output_path=args.output,
        )
        print(frame.head().to_string())
    else:  # pragma: no cover
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
