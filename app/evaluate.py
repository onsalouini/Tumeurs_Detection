import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import requests

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


# Handles common filename prefixes found in public brain MRI datasets.
DEFAULT_PREFIX_MAP = {
    "te-gl": "glioma",
    "te-me": "meningioma",
    "te-aug-me": "meningioma",
    "te-no": "notumor",
    "te-pi": "pituitary",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch evaluation for the tumor detection API."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Dataset directory. Prefer structure data_dir/class_name/*.jpg",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/predict",
        help="Prediction endpoint URL",
    )
    parser.add_argument(
        "--class-names-json",
        default="models/class_names.json",
        help="Path to class_names.json",
    )
    parser.add_argument(
        "--output",
        default="app/eval_report.json",
        help="Path to save JSON report",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Timeout per image request in seconds",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan images inside class folders",
    )
    return parser.parse_args()


def load_class_names(path_str):
    path = Path(path_str)
    if not path.exists():
        return DEFAULT_CLASS_NAMES

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return data
    return DEFAULT_CLASS_NAMES


def discover_images(data_dir, class_names, recursive=False):
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    class_dirs = [d for d in root.iterdir() if d.is_dir() and d.name in class_names]
    items = []

    if class_dirs:
        for cls_dir in class_dirs:
            iterator = cls_dir.rglob("*") if recursive else cls_dir.glob("*")
            for p in iterator:
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    items.append((p, cls_dir.name))
        return items

    # Fallback for flat folders: infer class from filename prefix.
    iterator = root.rglob("*") if recursive else root.glob("*")
    for p in iterator:
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        low_name = p.name.lower()
        inferred = None
        for prefix, class_name in DEFAULT_PREFIX_MAP.items():
            if low_name.startswith(prefix):
                inferred = class_name
                break
        if inferred:
            items.append((p, inferred))

    return items


def predict_image(image_path, api_url, timeout_s):
    with image_path.open("rb") as f:
        files = {"file": (image_path.name, f, "application/octet-stream")}
        resp = requests.post(api_url, files=files, data={"modality": "brain_mri"}, timeout=timeout_s)

    try:
        payload = resp.json()
    except Exception:
        payload = {"raw": resp.text}

    return resp.status_code, payload


def safe_div(a, b):
    return a / b if b else 0.0


def build_confusion_matrix(labels, y_true, y_pred):
    idx = {name: i for i, name in enumerate(labels)}
    size = len(labels)
    cm = [[0 for _ in range(size)] for _ in range(size)]

    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1

    return cm


def per_class_metrics(labels, cm):
    metrics = {}
    total = sum(sum(row) for row in cm)

    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels)) if r != i)
        fn = sum(cm[i][c] for c in range(len(labels)) if c != i)
        tn = total - tp - fp - fn

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        support = sum(cm[i])

        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    return metrics


def evaluate(data_items, api_url, labels, timeout_s):
    y_true = []
    y_pred = []
    errors = []
    status_counter = Counter()
    skipped_counter = Counter()

    for image_path, true_label in data_items:
        try:
            code, payload = predict_image(image_path, api_url, timeout_s)
        except Exception as e:
            errors.append({
                "image": str(image_path),
                "error": str(e),
            })
            continue

        if code != 200:
            errors.append({
                "image": str(image_path),
                "http_status": code,
                "response": payload,
            })
            continue

        status = payload.get("status", "UNKNOWN")
        status_counter[status] += 1

        pred = payload.get("prediction")
        if not pred:
            skipped_counter["no_prediction"] += 1
            continue
        if pred not in labels:
            skipped_counter["out_of_vocab_prediction"] += 1
            continue
        if true_label not in labels:
            skipped_counter["out_of_vocab_true_label"] += 1
            continue

        y_true.append(true_label)
        y_pred.append(pred)

    total = len(data_items)
    evaluated = len(y_true)
    cm = build_confusion_matrix(labels, y_true, y_pred)
    correct = sum(cm[i][i] for i in range(len(labels)))
    accuracy = safe_div(correct, evaluated)

    class_stats = per_class_metrics(labels, cm)
    macro_p = safe_div(sum(v["precision"] for v in class_stats.values()), len(labels))
    macro_r = safe_div(sum(v["recall"] for v in class_stats.values()), len(labels))
    macro_f1 = safe_div(sum(v["f1"] for v in class_stats.values()), len(labels))

    report = {
        "summary": {
            "total_images_found": total,
            "evaluated_images": evaluated,
            "errors": len(errors),
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
        },
        "statuses": dict(status_counter),
        "skipped": dict(skipped_counter),
        "labels": labels,
        "confusion_matrix": cm,
        "per_class": class_stats,
        "errors": errors,
    }

    return report


def main():
    args = parse_args()
    labels = load_class_names(args.class_names_json)
    items = discover_images(args.data_dir, labels, recursive=args.recursive)

    if not items:
        raise SystemExit(
            "No labeled images found. Use class folders or supported filename prefixes."
        )

    report = evaluate(items, args.api_url, labels, args.timeout)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== Evaluation complete ===")
    print(f"Images found: {report['summary']['total_images_found']}")
    print(f"Images evaluated: {report['summary']['evaluated_images']}")
    print(f"Errors: {report['summary']['errors']}")
    print(f"Accuracy: {report['summary']['accuracy']}")
    print(f"Macro F1: {report['summary']['macro_f1']}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
