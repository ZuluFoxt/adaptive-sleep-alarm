#!/usr/bin/env python3
"""End-to-end sleep-stage and adaptive-alarm benchmark with wearable PPG heart rate.

Data source:
    PhysioNet sleep-accel v1.0.0 (Walch et al., SLEEP, 2019)
    https://physionet.org/content/sleep-accel/1.0.0/

Only labels and heart_rate text files are downloaded. The large motion and steps
directories are intentionally never accessed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_URL = "https://physionet.org/files/sleep-accel/1.0.0"
EPOCH_SEC = 30
CLASS_NAMES = ["Wake", "Light", "Deep", "REM"]
CLASS_SHORT = ["W", "Light", "Deep", "REM"]
STAGE_NAMES = ["Wake", "N1", "N2", "N3", "N4", "REM"]
STAGE_COLORS = ["#f4c95d", "#a8dadc", "#79b9a6", "#315b7d", "#1d3557", "#b06ab3"]
CLASS_COLORS = ["#f4c95d", "#78b7a5", "#315b7d", "#b06ab3"]
PROB_COLS = ["P_Wake", "P_Light", "P_Deep", "P_REM"]
STRATEGIES = [
    "Fixed",
    "WakeOnly",
    "Greedy_Light_k3",
    "Dynamic_Sym_k3",
    "Dynamic_Asym_Proposed",
]


@dataclass
class SubjectData:
    sid: str
    labels: pd.DataFrame
    hr: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a subject-wise, causal PPG sleep/alarm benchmark."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/sleep-accel-1.0.0")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Require labels/heart_rate files to already exist.",
    )
    parser.add_argument(
        "--force-download", action="store_true", help="Redownload all text files."
    )
    parser.add_argument(
        "--demo-subject",
        default=None,
        help="Prefer this subject for subplot B if it has a valid proposed trigger.",
    )
    return parser.parse_args()


def _requests_session():
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError as exc:
        raise RuntimeError(
            "缺少 requests。请先执行：python -m pip install requests"
        ) from exc

    retry = Retry(
        total=7,
        connect=7,
        read=7,
        status=7,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(("GET",)),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {"User-Agent": "Codex-sleep-accel-benchmark/1.0 (academic course project)"}
    )
    return session


def _directory_ids(session, folder: str, suffix: str) -> list[str]:
    response = session.get(f"{BASE_URL}/{folder}/", timeout=(20, 60))
    response.raise_for_status()
    pattern = rf'href="([^"]+_{re.escape(suffix)})"'
    names = re.findall(pattern, response.text)
    return sorted({int(name.split("_", 1)[0]) for name in names}, key=str)


def download_data(data_dir: Path, skip_download: bool, force: bool) -> None:
    labels_dir = data_dir / "labels"
    heart_dir = data_dir / "heart_rate"
    labels_dir.mkdir(parents=True, exist_ok=True)
    heart_dir.mkdir(parents=True, exist_ok=True)

    expected_labels = data_dir / "download_manifest.json"
    if skip_download:
        if not expected_labels.exists():
            raise FileNotFoundError(
                f"{expected_labels} 不存在；不能使用 --skip-download。"
            )
        return

    session = _requests_session()
    print("读取 PhysioNet 目录清单（仅 labels 与 heart_rate）……", flush=True)
    label_ids = _directory_ids(session, "labels", "labeled_sleep.txt")
    heart_ids = _directory_ids(session, "heart_rate", "heartrate.txt")
    ids = sorted(set(label_ids) & set(heart_ids), key=str)
    if len(ids) != 31:
        raise RuntimeError(f"期望 31 名受试者，目录清单只匹配到 {len(ids)} 名。")

    expected = []
    for sid in ids:
        expected.append(
            (f"{BASE_URL}/labels/{sid}_labeled_sleep.txt", labels_dir)
        )
        expected.append((f"{BASE_URL}/heart_rate/{sid}_heartrate.txt", heart_dir))

    existing_bytes = sum(
        path.stat().st_size
        for path in list(labels_dir.glob("*.txt")) + list(heart_dir.glob("*.txt"))
        if path.exists()
    )
    if not force and all((folder / Path(url).name).exists() for url, folder in expected):
        print(
            f"数据已完整缓存：{len(ids)} 名受试者，本地文本约 "
            f"{existing_bytes / 1024 / 1024:.2f} MB。",
            flush=True,
        )
    else:
        print(
            "开始下载 31 组 labels/heart_rate（预计约 5 MB，正常网络约 20–30 秒）……",
            flush=True,
        )
        started = time.monotonic()
        downloaded = 0
        for url, folder in expected:
            destination = folder / Path(url).name
            if destination.exists() and not force and destination.stat().st_size > 0:
                continue
            temp = destination.with_suffix(destination.suffix + ".part")
            last_error = None
            for attempt in range(1, 8):
                try:
                    with session.get(url, stream=True, timeout=(20, 90)) as response:
                        response.raise_for_status()
                        with temp.open("wb") as handle:
                            for chunk in response.iter_content(chunk_size=64 * 1024):
                                if chunk:
                                    handle.write(chunk)
                    temp.replace(destination)
                    downloaded += 1
                    last_error = None
                    break
                except Exception as exc:  # transient SSL/S3 failures are retried
                    last_error = exc
                    temp.unlink(missing_ok=True)
                    time.sleep(min(2.0 * attempt, 8.0))
            if last_error is not None:
                raise RuntimeError(f"下载失败：{url}\n{last_error}") from last_error
        elapsed = time.monotonic() - started
        total_bytes = sum(
            (folder / Path(url).name).stat().st_size for url, folder in expected
        )
        print(
            f"下载完成：{downloaded} 个文件，约 {total_bytes / 1024 / 1024:.2f} MB，"
            f"耗时 {elapsed:.1f} 秒。",
            flush=True,
        )

    actual_ids = sorted(
        int(path.name.split("_", 1)[0]) for path in labels_dir.glob("*_labeled_sleep.txt")
    )
    manifest = {
        "source": f"{BASE_URL}/",
        "subjects": [str(sid) for sid in ids],
        "label_files": len(label_ids),
        "heart_rate_files": len(heart_ids),
        "downloaded_directories": ["labels", "heart_rate"],
    }
    expected_labels.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"受试者清单：{', '.join(str(i) for i in actual_ids)}", flush=True)


def map_stage(stage_id: int) -> float:
    mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3}
    return float(mapping.get(int(stage_id), np.nan))


def safe_stat(values: np.ndarray, function, default=np.nan) -> float:
    if values.size == 0:
        return float(default)
    result = function(values)
    return float(result) if np.isfinite(result) else float(default)


def rmssd_proxy(values: np.ndarray) -> float:
    if values.size < 2:
        return np.nan
    return float(np.sqrt(np.mean(np.diff(values) ** 2)))


def mean_abs_diff(values: np.ndarray) -> float:
    if values.size < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(values))))


def logistic_squash(value: float, scale: float = 2.0) -> float:
    if not np.isfinite(value):
        return 0.5
    value = np.clip(value / scale, -30, 30)
    return float(1.0 / (1.0 + np.exp(-value)))


def extract_causal_features(subject: SubjectData) -> pd.DataFrame:
    labels = subject.labels.sort_values("timestamp_sec").reset_index(drop=True)
    hr = subject.hr.sort_values("timestamp_sec").reset_index(drop=True)
    hr = hr[
        (hr["timestamp_sec"] >= -1800)
        & (hr["timestamp_sec"] <= labels["timestamp_sec"].max() + EPOCH_SEC)
        & hr["bpm"].between(25, 220)
    ].reset_index(drop=True)
    ts = hr["timestamp_sec"].to_numpy(dtype=float)
    bpm = hr["bpm"].to_numpy(dtype=float)

    def values_between(start: float, end: float) -> np.ndarray:
        left = np.searchsorted(ts, start, side="left")
        right = np.searchsorted(ts, end, side="left")
        return bpm[left:right]

    records: list[dict] = []
    for _, row in labels.iterrows():
        epoch_start = float(row["timestamp_sec"])
        stage_id = int(row["stage_id"])
        target = map_stage(stage_id)
        if not np.isfinite(target):
            continue

        current_start = epoch_start - 60
        current_end = epoch_start + EPOCH_SEC
        current = values_between(current_start, current_end)
        recent_120 = values_between(epoch_start - 120, current_end)
        is_unknown = recent_120.size < 3 or current.size == 0

        record: dict[str, float | int | str] = {
            "sid": subject.sid,
            "timestamp_sec": epoch_start,
            "stage_id_original": stage_id,
            "stage_id": int(target),
            "is_unknown": int(is_unknown),
        }

        record["hr_mean_90s"] = safe_stat(current, np.mean)
        record["hr_std_90s"] = safe_stat(
            current, lambda x: np.std(x, ddof=1) if x.size > 1 else 0.0
        )
        record["hr_range_90s"] = (
            safe_stat(current, np.ptp) if current.size else np.nan
        )
        record["sample_density_90s_per_min"] = current.size / 1.5

        for minutes in (2, 5, 10):
            window = values_between(epoch_start - minutes * 60, current_end)
            suffix = f"{minutes}m"
            record[f"hr_mean_{suffix}"] = safe_stat(window, np.mean)
            record[f"hr_std_{suffix}"] = safe_stat(
                window, lambda x: np.std(x, ddof=1) if x.size > 1 else 0.0
            )
            record[f"rmssd_{suffix}"] = rmssd_proxy(window)
            record[f"mean_abs_diff_{suffix}"] = mean_abs_diff(window)

        expanding = values_between(0, current_end)
        rolling_30m = values_between(epoch_start - 1800, current_end)
        current_mean = record["hr_mean_90s"]
        current_std = record["hr_std_90s"]
        for baseline_name, baseline in (
            ("expanding", expanding),
            ("rolling30m", rolling_30m),
        ):
            base_mean = safe_stat(baseline, np.mean, current_mean)
            base_std = safe_stat(
                baseline, lambda x: np.std(x, ddof=1) if x.size > 1 else 0.0, 0.0
            )
            base_std = max(base_std, 1.0)
            z_value = (
                (current_mean - base_mean) / base_std
                if np.isfinite(current_mean) and np.isfinite(base_mean)
                else np.nan
            )
            percentile = (
                float(np.mean(baseline <= current_mean))
                if baseline.size and np.isfinite(current_mean)
                else np.nan
            )
            record[f"hr_z_{baseline_name}"] = z_value
            record[f"hr_percentile_{baseline_name}"] = percentile
            record[f"baseline_std_{baseline_name}"] = base_std

        arousal_window = values_between(epoch_start - 90, current_end)
        if arousal_window.size >= 3:
            x = np.arange(arousal_window.size, dtype=float)
            slope_per_sample = float(np.polyfit(x, arousal_window, 1)[0])
            slope_per_min = slope_per_sample * max(1, arousal_window.size)
            jump = (
                float(np.max(current) - np.median(values_between(epoch_start - 300, epoch_start)))
                if current.size
                and values_between(epoch_start - 300, epoch_start).size
                else np.nan
            )
            scale = max(float(record["hr_std_5m"]), 1.0)
            slope_z = slope_per_min / scale
            jump_z = jump / scale if np.isfinite(jump) else np.nan
            record["hr_slope_90s_bpm_per_min"] = slope_per_min
            record["hr_jump_30s_bpm"] = jump
            record["arousal_index"] = (
                0.5 * logistic_squash(slope_z)
                + 0.5 * logistic_squash(jump_z)
            )
        else:
            record["hr_slope_90s_bpm_per_min"] = np.nan
            record["hr_jump_30s_bpm"] = np.nan
            record["arousal_index"] = np.nan

        record["time_since_record_start_hours"] = max(0.0, epoch_start / 3600.0)
        records.append(record)

    features = pd.DataFrame.from_records(records)
    features = features.sort_values(["timestamp_sec"]).reset_index(drop=True)
    return features


def load_subjects(data_dir: Path) -> list[SubjectData]:
    subjects = []
    label_files = sorted((data_dir / "labels").glob("*_labeled_sleep.txt"))
    for label_path in label_files:
        sid = label_path.name.split("_", 1)[0]
        heart_path = data_dir / "heart_rate" / f"{sid}_heartrate.txt"
        if not heart_path.exists():
            continue
        labels = pd.read_csv(
            label_path,
            sep=r"\s+",
            header=None,
            names=["timestamp_sec", "stage_id"],
        )
        hr = pd.read_csv(
            heart_path,
            sep=",",
            header=None,
            names=["timestamp_sec", "bpm"],
        )
        labels = labels.apply(pd.to_numeric, errors="coerce").dropna()
        hr = hr.apply(pd.to_numeric, errors="coerce").dropna()
        subjects.append(SubjectData(sid=sid, labels=labels, hr=hr))
    if len(subjects) != 31:
        raise RuntimeError(f"应加载 31 名受试者，实际得到 {len(subjects)} 名。")
    return subjects


def build_models() -> dict[str, Pipeline]:
    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=42
                ),
            ),
        ]
    )
    tree = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=220,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    return {"LogisticRegression": logistic, "HistGradientBoosting": tree}


def metric_values(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Iterable[int]
) -> tuple[float, float, float]:
    labels = list(labels)
    return (
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        accuracy_score(y_true, y_pred),
        cohen_kappa_score(y_true, y_pred, labels=labels),
    )


def remap_for_task(values: np.ndarray, dimension: int) -> np.ndarray:
    if dimension == 4:
        return values.astype(int)
    if dimension == 3:
        return np.where(values == 0, 0, np.where(values == 3, 2, 1)).astype(int)
    if dimension == 2:
        return np.where(values <= 1, 0, 1).astype(int)
    raise ValueError(f"Unsupported dimension: {dimension}")


def train_subject_wise_cv(
    features: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    feature_columns = [
        column
        for column in features.columns
        if column
        not in {
            "sid",
            "timestamp_sec",
            "stage_id_original",
            "stage_id",
            "is_unknown",
        }
    ]
    known = features[features["is_unknown"] == 0].copy()
    X = known[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = known["stage_id"].astype(int).to_numpy()
    groups = known["sid"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)

    all_metrics: list[dict] = []
    oof_parts: list[pd.DataFrame] = []
    classifier_metrics: dict[str, float] = {}

    for model_name, model in build_models().items():
        print(f"训练 Subject-wise 5-Fold：{model_name}", flush=True)
        oof_frame = known.copy()
        oof_frame["classifier"] = model_name
        for column in PROB_COLS:
            oof_frame[column] = np.nan
        oof_frame["predicted_stage"] = -1

        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X, y, groups=groups), start=1
        ):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test).astype(int)
            probabilities = model.predict_proba(X_test)
            full_probabilities = np.zeros((len(test_idx), 4), dtype=float)
            for probability_col, class_value in enumerate(model.classes_):
                full_probabilities[:, int(class_value)] = probabilities[
                    :, probability_col
                ]

            frame_positions = oof_frame.index.to_numpy()[test_idx]
            oof_frame.loc[frame_positions, "predicted_stage"] = predictions
            oof_frame.loc[frame_positions, PROB_COLS] = full_probabilities

            for dimension, labels in ((4, [0, 1, 2, 3]), (3, [0, 1, 2]), (2, [0, 1])):
                task_true = remap_for_task(y_test, dimension)
                task_pred = remap_for_task(predictions, dimension)
                macro_f1, accuracy, kappa = metric_values(
                    task_true, task_pred, labels
                )
                all_metrics.append(
                    {
                        "classifier": model_name,
                        "scope": f"fold_{fold}",
                        "task": f"{dimension}class",
                        "n_epochs": len(test_idx),
                        "macro_f1": macro_f1,
                        "accuracy": accuracy,
                        "cohen_kappa": kappa,
                        "unknown_ratio": float(
                            features.loc[features["sid"].isin(np.unique(groups[test_idx])), "is_unknown"].mean()
                        ),
                    }
                )

        valid_oof = oof_frame["predicted_stage"] >= 0
        for dimension, labels in ((4, [0, 1, 2, 3]), (3, [0, 1, 2]), (2, [0, 1])):
            task_true = remap_for_task(
                oof_frame.loc[valid_oof, "stage_id"].to_numpy(), dimension
            )
            task_pred = remap_for_task(
                oof_frame.loc[valid_oof, "predicted_stage"].to_numpy(), dimension
            )
            macro_f1, accuracy, kappa = metric_values(task_true, task_pred, labels)
            all_metrics.append(
                {
                    "classifier": model_name,
                    "scope": "out_of_fold_overall",
                    "task": f"{dimension}class",
                    "n_epochs": int(valid_oof.sum()),
                    "macro_f1": macro_f1,
                    "accuracy": accuracy,
                    "cohen_kappa": kappa,
                    "unknown_ratio": float(features["is_unknown"].mean()),
                }
            )
            if dimension == 4:
                classifier_metrics[model_name] = float(macro_f1)

        all_unknown = features.copy()
        all_unknown["classifier"] = model_name
        prediction_columns = oof_frame[
            [
                "sid",
                "timestamp_sec",
                "stage_id_original",
                "stage_id",
                "is_unknown",
                "predicted_stage",
                *PROB_COLS,
            ]
        ]
        all_unknown = all_unknown.merge(
            prediction_columns,
            on=[
                "sid",
                "timestamp_sec",
                "stage_id_original",
                "stage_id",
                "is_unknown",
            ],
            how="left",
        )
        all_unknown.loc[
            all_unknown["is_unknown"].eq(1), "predicted_stage"
        ] = -1
        oof_parts.append(all_unknown)

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(output_dir / "ppg_staging_metrics.csv", index=False)
    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(output_dir / "ppg_oof_predictions.csv", index=False)
    selected = max(classifier_metrics, key=classifier_metrics.get)
    return metrics, oof, selected


def calibrate_thresholds(subject_oof: pd.DataFrame) -> tuple[float, float]:
    subset = subject_oof[
        subject_oof["stage_id"].isin([0, 1])
        & subject_oof["P_Wake"].notna()
    ]
    if subset.empty:
        return 0.55, 0.15
    scores = alarm_score(subset)
    theta_max = float(np.nanquantile(scores, 0.85))
    theta_min = float(np.nanquantile(scores, 0.25))
    theta_max = min(theta_max, 1.4)
    theta_min = max(min(theta_min, theta_max - 0.05), -0.8)
    return theta_max, theta_min


def alarm_score(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["P_Wake"].to_numpy(dtype=float)
        + 0.6 * frame["P_Light"].to_numpy(dtype=float)
        - 0.4 * frame["P_REM"].to_numpy(dtype=float)
        - frame["P_Deep"].to_numpy(dtype=float)
        + 0.20 * frame["arousal_index"].fillna(0.5).to_numpy(dtype=float)
    )


def causal_window_mean(values: np.ndarray, index: int, start: int, k: int) -> float:
    left = max(start, index - k + 1)
    local = values[left : index + 1]
    local = local[np.isfinite(local)]
    return float(np.mean(local)) if local.size else np.nan


def simulate_strategy(
    frame: pd.DataFrame,
    scores: np.ndarray,
    end_index: int,
    window_epochs: int,
    strategy: str,
    theta_max: float,
    theta_min: float,
) -> dict:
    start_index = max(0, end_index - window_epochs + 1)
    trigger_index = None

    if strategy == "Fixed":
        trigger_index = end_index
    elif strategy == "WakeOnly":
        consecutive = 0
        for index in range(start_index, end_index + 1):
            valid = (
                frame.iloc[index]["is_unknown"] == 0
                and frame.iloc[index]["predicted_stage"] == 0
            )
            consecutive = consecutive + 1 if valid else 0
            if consecutive >= 2:
                trigger_index = index
                break
    elif strategy == "Greedy_Light_k3":
        for index in range(start_index, end_index + 1):
            if frame.iloc[index]["is_unknown"] != 0:
                continue
            trailing = causal_window_mean(scores, index, start_index, 3)
            classes = frame.iloc[max(start_index, index - 2) : index + 1][
                "predicted_stage"
            ].to_numpy(dtype=int)
            three_safe = classes.size == 3 and np.all(np.isin(classes, [0, 1]))
            if np.isfinite(trailing) and (trailing >= 0.15 or three_safe):
                trigger_index = index
                break
    elif strategy == "Dynamic_Sym_k3":
        for index in range(start_index, end_index + 1):
            if frame.iloc[index]["is_unknown"] != 0:
                continue
            trailing = causal_window_mean(scores, index, start_index, 3)
            tau = max(0.0, min(1.0, (end_index - index) / max(window_epochs, 1)))
            threshold = theta_min + (theta_max - theta_min) * tau**1.15
            if np.isfinite(trailing) and trailing >= threshold:
                trigger_index = index
                break
    elif strategy == "Dynamic_Asym_Proposed":
        filtered = np.nan
        for index in range(start_index, end_index + 1):
            if frame.iloc[index]["is_unknown"] != 0 or not np.isfinite(scores[index]):
                filtered = np.nan
                continue
            if not np.isfinite(filtered):
                filtered = float(scores[index])
            else:
                alpha = 0.40 if scores[index] >= filtered else 0.85
                filtered = alpha * scores[index] + (1 - alpha) * filtered
            tau = max(0.0, min(1.0, (end_index - index) / max(window_epochs, 1)))
            threshold = theta_min + (theta_max - theta_min) * tau**1.15
            if filtered >= threshold:
                trigger_index = index
                break
    else:
        raise ValueError(strategy)

    forced = trigger_index is None
    if forced:
        trigger_index = end_index
    true_stage = int(frame.iloc[trigger_index]["stage_id"])
    return {
        "strategy": strategy,
        "window_epochs": window_epochs,
        "window_minutes": window_epochs // 2,
        "t_end_index": end_index,
        "t_end_sec": float(frame.iloc[end_index]["timestamp_sec"]),
        "trigger_index": trigger_index,
        "trigger_sec": float(frame.iloc[trigger_index]["timestamp_sec"]),
        "trigger_predicted_stage": int(frame.iloc[trigger_index]["predicted_stage"]),
        "trigger_true_stage": true_stage,
        "triggered_unknown": int(frame.iloc[trigger_index]["is_unknown"]),
        "forced": int(forced),
        "early_minutes": max(
            0.0, (float(frame.iloc[end_index]["timestamp_sec"]) - float(frame.iloc[trigger_index]["timestamp_sec"])) / 60.0
        ),
    }


def run_alarm_benchmark(
    oof: pd.DataFrame, selected_classifier: str, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    selected = oof[oof["classifier"] == selected_classifier].copy()
    selected["arousal_index"] = selected["arousal_index"].fillna(0.5)
    selected["score"] = alarm_score(selected)
    results = []
    cutoff_results = []
    demo_candidates = []

    for sid, subject_frame in selected.groupby("sid", sort=True):
        subject_frame = subject_frame.sort_values("timestamp_sec").reset_index(drop=True)
        if subject_frame["P_Deep"].notna().sum() == 0:
            continue
        known_positions = np.flatnonzero(
            subject_frame["stage_id"].notna().to_numpy()
            & subject_frame["predicted_stage"].ge(0).to_numpy()
        )
        if known_positions.size < 121:
            continue
        last_index = int(known_positions[-1])
        theta_max, theta_min = calibrate_thresholds(subject_frame)
        scores = subject_frame["score"].to_numpy(dtype=float)
        end_candidates = list(range(max(120, last_index - 120), last_index + 1, 2))

        for window_minutes in (20, 30):
            window_epochs = window_minutes * 2
            for end_index in end_candidates:
                if end_index + 1 < window_epochs:
                    continue
                for strategy in STRATEGIES:
                    result = simulate_strategy(
                        subject_frame,
                        scores,
                        end_index,
                        window_epochs,
                        strategy,
                        theta_max,
                        theta_min,
                    )
                    result.update(
                        {
                            "sid": sid,
                            "theta_max": theta_max,
                            "theta_min": theta_min,
                        }
                    )
                    results.append(result)
                    bad_avoided = 0
                    if result["trigger_index"] < end_index:
                        stages_after = subject_frame.iloc[
                            result["trigger_index"] + 1 : end_index + 1
                        ]["stage_id"]
                        bad_avoided = int(stages_after.isin([2, 3]).sum())
                    if result["strategy"] == "Dynamic_Asym_Proposed":
                        demo_candidates.append(
                            {
                                "sid": sid,
                                "window_minutes": window_minutes,
                                "end_index": end_index,
                                "bad_avoided": bad_avoided,
                                "early_minutes": result["early_minutes"],
                                "true_stage": result["trigger_true_stage"],
                            }
                        )

        for strategy in STRATEGIES:
            cutoff_results.append(
                {
                    "sid": sid,
                    "strategy": strategy,
                    "true_last_stage": int(subject_frame.iloc[last_index]["stage_id"]),
                    "last_sec": float(
                        subject_frame.iloc[last_index]["timestamp_sec"]
                    ),
                }
            )

    alarm_results = pd.DataFrame(results)
    alarm_results.to_csv(output_dir / "ppg_alarm_end2end_results.csv", index=False)
    summary = (
        alarm_results.assign(
            is_n3=alarm_results["trigger_true_stage"].eq(2),
            is_rem=alarm_results["trigger_true_stage"].eq(3),
            is_bad=alarm_results["trigger_true_stage"].isin([2, 3]),
            is_safe=alarm_results["trigger_true_stage"].isin([0, 1]),
        )
        .groupby(["window_minutes", "strategy"], sort=False)
        .agg(
            n_trials=("sid", "size"),
            n_subjects=("sid", "nunique"),
            n3_wake_rate=("is_n3", "mean"),
            rem_wake_rate=("is_rem", "mean"),
            total_bad_wake_rate=("is_bad", "mean"),
            safe_wake_rate=("is_safe", "mean"),
            mean_early_minutes=("early_minutes", "mean"),
            forced_rate=("forced", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "ppg_alarm_summary.csv", index=False)

    demo_stats = {}
    if demo_candidates:
        demo_frame = pd.DataFrame(demo_candidates)
        demo_frame = demo_frame[demo_frame["true_stage"].isin([0, 1])]
        if not demo_frame.empty:
            median_early = float(demo_frame["early_minutes"].median())
            demo_frame["typicality_gap"] = (
                demo_frame["early_minutes"] - median_early
            ).abs()
            with_avoidance = demo_frame[demo_frame["bad_avoided"] > 0]
            if not with_avoidance.empty:
                demo_frame = with_avoidance
            demo_frame = demo_frame.sort_values(
                ["typicality_gap", "bad_avoided"],
                ascending=[True, False],
            )
            chosen = demo_frame.iloc[0]
            demo_stats = chosen.to_dict()
    return alarm_results, summary, demo_stats


def choose_chinese_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]:
        if candidate in available:
            return candidate
    print("警告：未找到中文字体，图中中文可能显示方框。", flush=True)
    return "DejaVu Sans"


def plot_benchmark(
    features: pd.DataFrame,
    oof: pd.DataFrame,
    selected_classifier: str,
    alarm_results: pd.DataFrame,
    summary: pd.DataFrame,
    demo_stats: dict,
    raw_subjects: dict[str, SubjectData],
    output_path: Path,
    preferred_demo_subject: str | None,
) -> None:
    font = choose_chinese_font()
    plt.rcParams.update(
        {
            "font.family": font,
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    selected = oof[oof["classifier"] == selected_classifier].copy()
    valid = selected["predicted_stage"].ge(0)
    cm = confusion_matrix(
        selected.loc[valid, "stage_id"].astype(int),
        selected.loc[valid, "predicted_stage"].astype(int),
        labels=[0, 1, 2, 3],
    )
    cm_percent = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100

    figure = plt.figure(figsize=(15.5, 10.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 1.15])
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, :])

    image = ax_a.imshow(cm_percent, cmap="Blues", vmin=0, vmax=100)
    ax_a.set(
        xticks=np.arange(4),
        yticks=np.arange(4),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="预测分期",
        ylabel="PSG 金标准",
        title=f"(A) Subject-wise 5 折交叉验证混淆矩阵（{selected_classifier}）",
    )
    for row in range(4):
        for column in range(4):
            value = cm_percent[row, column]
            color = "white" if value > 58 else "#172033"
            ax_a.text(
                column,
                row,
                f"{value:.1f}%\n(n={cm[row, column]})",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    colorbar = figure.colorbar(image, ax=ax_a, fraction=0.046, pad=0.04)
    colorbar.set_label("行归一化比例 (%)")

    # If the automatically selected demonstration has no trigger, choose from trials.
    if preferred_demo_subject:
        preferred = [
            row
            for row in [
                demo_stats,
            ]
            if row and str(row.get("sid")) == str(preferred_demo_subject)
        ]
        if preferred:
            demo_stats = preferred[0]

    if not demo_stats:
        proposed = alarm_results[
            (alarm_results["strategy"] == "Dynamic_Asym_Proposed")
            & (alarm_results["window_minutes"] == 20)
            & alarm_results["trigger_true_stage"].isin([0, 1])
        ].sort_values("early_minutes", ascending=False)
        if proposed.empty:
            demo_stats = {
                "sid": str(alarm_results.iloc[0]["sid"]),
                "window_minutes": 20,
                "end_index": int(alarm_results.iloc[0]["t_end_index"]),
                "early_minutes": float(alarm_results.iloc[0]["early_minutes"]),
                "true_stage": int(alarm_results.iloc[0]["trigger_true_stage"]),
                "bad_avoided": 0,
            }
        else:
            demo_stats = proposed.iloc[0].to_dict()

    sid = str(demo_stats["sid"])
    if sid.endswith(".0"):
        sid = sid[:-2]
    selected_sid = selected["sid"].astype(str).str.replace(r"\.0$", "", regex=True)
    subject_frame = selected[selected_sid == sid].sort_values(
        "timestamp_sec"
    ).reset_index(drop=True)
    subject_frame["arousal_index"] = subject_frame["arousal_index"].fillna(0.5)
    subject_frame["score"] = alarm_score(subject_frame)
    theta_max, theta_min = calibrate_thresholds(subject_frame)
    scores = subject_frame["score"].to_numpy(dtype=float)

    end_index = min(int(demo_stats["end_index"]), len(subject_frame) - 1)
    window_minutes = int(demo_stats.get("window_minutes", 20))
    window_epochs = window_minutes * 2
    start_index = max(0, end_index - 59)
    filtered_scores = np.full(end_index - start_index + 1, np.nan)
    thresholds = np.zeros(end_index - start_index + 1)
    ewma = np.nan
    for offset, index in enumerate(range(start_index, end_index + 1)):
        tau = max(0.0, min(1.0, (end_index - index) / window_epochs))
        thresholds[offset] = theta_min + (theta_max - theta_min) * tau**1.15
        if subject_frame.iloc[index]["is_unknown"] != 0 or not np.isfinite(scores[index]):
            ewma = np.nan
            continue
        if not np.isfinite(ewma):
            ewma = float(scores[index])
        else:
            alpha = 0.40 if scores[index] >= ewma else 0.85
            ewma = alpha * scores[index] + (1 - alpha) * ewma
        filtered_scores[offset] = ewma

    result_sid = alarm_results["sid"].astype(str).str.replace(
        r"\.0$", "", regex=True
    )
    proposed_rows = alarm_results[
        (result_sid == sid)
        & (alarm_results["strategy"] == "Dynamic_Asym_Proposed")
        & (alarm_results["window_minutes"] == window_minutes)
        & (alarm_results["t_end_index"] == end_index)
    ]
    if proposed_rows.empty:
        trigger_index = end_index
        forced = True
    else:
        trigger_index = int(proposed_rows.iloc[0]["trigger_index"])
        forced = bool(proposed_rows.iloc[0]["forced"])

    x_minutes = (
        subject_frame.iloc[start_index : end_index + 1]["timestamp_sec"].to_numpy()
        - subject_frame.iloc[end_index]["timestamp_sec"]
    ) / 60.0
    for index in range(start_index, end_index + 1):
        stage = int(subject_frame.iloc[index]["stage_id"])
        ax_b.axvspan(
            x_minutes[index - start_index] - 0.25,
            x_minutes[index - start_index] + 0.25,
            ymin=0.88,
            ymax=1.0,
            color=STAGE_COLORS[stage],
            alpha=0.9,
            linewidth=0,
        )

    score_axis = ax_b
    score_axis.plot(x_minutes, filtered_scores, color="#173f5f", lw=2.0, label=r"$\bar{S}_{asym}$")
    score_axis.plot(
        x_minutes,
        thresholds,
        color="#d1495b",
        lw=1.45,
        ls="--",
        label=r"动态阈值 $\Theta(t)$",
    )
    score_axis.axvline(0, color="#333333", lw=1.0, ls=":")
    trigger_minutes = (
        subject_frame.iloc[trigger_index]["timestamp_sec"]
        - subject_frame.iloc[end_index]["timestamp_sec"]
    ) / 60.0
    trigger_stage = int(subject_frame.iloc[trigger_index]["stage_id"])
    trigger_text = "强制兜底" if forced else f"建议触发\n{trigger_minutes:.0f} min"
    score_axis.axvline(
        trigger_minutes,
        color="#2a9d8f",
        lw=2.2,
        label=f"{trigger_text}（真实{STAGE_NAMES[trigger_stage]}）",
    )
    score_axis.scatter(
        [trigger_minutes],
        [filtered_scores[max(0, trigger_index - start_index)]],
        s=44,
        color="#2a9d8f",
        edgecolor="white",
        zorder=6,
    )
    score_axis.set(
        xlabel=f"距离最晚闹钟 $T_{{end}}$ 的分钟数（$W={window_minutes}$ min）",
        ylabel=r"唤醒适宜度得分 / 阈值",
        title=f"(B) {sid} 号受试者清晨真实 PPG 与自适应触发轨迹",
        xlim=(x_minutes[0], 0.5),
    )
    score_axis.grid(alpha=0.2)

    hr_axis = score_axis.twinx()
    raw_hr = raw_subjects[sid].hr
    raw_hr = raw_hr[
        raw_hr["timestamp_sec"].between(
            subject_frame.iloc[start_index]["timestamp_sec"],
            subject_frame.iloc[end_index]["timestamp_sec"] + EPOCH_SEC,
        )
        & raw_hr["bpm"].between(25, 220)
    ]
    raw_x = (
        raw_hr["timestamp_sec"] - subject_frame.iloc[end_index]["timestamp_sec"]
    ) / 60.0
    hr_axis.plot(raw_x, raw_hr["bpm"], color="#e76f51", lw=0.8, alpha=0.7)
    hr_axis.set_ylabel("PPG 心率 (bpm)", color="#a9442f")
    hr_axis.tick_params(axis="y", colors="#a9442f")
    hr_axis.spines["right"].set_color("#e76f51")
    lines, labels = score_axis.get_legend_handles_labels()
    score_axis.legend(lines, labels, loc="lower left", framealpha=0.92)
    stage_patches = [
        Patch(facecolor=STAGE_COLORS[index], label=STAGE_NAMES[index])
        for index in range(4)
    ] + [Patch(facecolor=STAGE_COLORS[5], label="REM")]
    score_axis.legend(
        handles=lines + stage_patches,
        labels=labels + [patch.get_label() for patch in stage_patches],
        loc="lower left",
        ncol=2,
        framealpha=0.92,
    )

    strategies = STRATEGIES
    x_positions = np.arange(len(strategies))
    width = 0.34
    bars = {}
    for window_minutes, offset, alpha in ((20, -width / 2, 1.0), (30, width / 2, 0.62)):
        subset = summary[summary["window_minutes"] == window_minutes].set_index("strategy")
        n3 = [float(subset.loc[name, "n3_wake_rate"]) * 100 for name in strategies]
        rem = [float(subset.loc[name, "rem_wake_rate"]) * 100 for name in strategies]
        bottom = ax_c.bar(
            x_positions + offset,
            n3,
            width,
            color="#315b7d",
            alpha=alpha,
            edgecolor="white",
            label=f"W={window_minutes}m · N3",
        )
        top = ax_c.bar(
            x_positions + offset,
            rem,
            width,
            bottom=n3,
            color="#b06ab3",
            alpha=alpha,
            edgecolor="white",
            hatch="//" if window_minutes == 30 else None,
            label=f"W={window_minutes}m · REM",
        )
        bars[window_minutes] = (bottom, top)
    ax_c.set(
        xticks=x_positions,
        xticklabels=[
            "Fixed",
            "WakeOnly",
            "Greedy\nLight k=3",
            "Dynamic\nSym k=3",
            "Dynamic\nAsym（提出）",
        ],
        ylabel="不良唤醒率 (%)",
        title="(C) 全受试者醒前 60 分钟端点遍历：不良唤醒构成与平均提前时长",
        xlabel="唤醒决策策略",
    )
    ax_c.grid(axis="y", alpha=0.2)
    ax_c.set_ylim(0, max(50, ax_c.get_ylim()[1] * 1.15))

    early_axis = ax_c.twinx()
    for window_minutes, color, linestyle, marker in (
        (20, "#d1495b", "-", "o"),
        (30, "#e9a03b", "--", "s"),
    ):
        subset = summary[summary["window_minutes"] == window_minutes].set_index(
            "strategy"
        )
        early = [subset.loc[name, "mean_early_minutes"] for name in strategies]
        early_axis.plot(
            x_positions,
            early,
            color=color,
            linestyle=linestyle,
            marker=marker,
            lw=1.8,
            ms=5,
            label=f"平均提前 {window_minutes}m 窗口",
        )
    early_axis.set_ylabel("平均提前唤醒时长 ΔT (min)", color="#7a2532")
    early_axis.tick_params(axis="y", colors="#7a2532")

    handles, labels = ax_c.get_legend_handles_labels()
    early_handles, early_labels = early_axis.get_legend_handles_labels()
    ax_c.legend(
        handles + early_handles,
        labels + early_labels,
        loc="upper right",
        ncol=2,
        framealpha=0.93,
    )
    figure.suptitle(
        "基于可穿戴 PPG 的因果睡眠分期与自适应唤醒端到端基准",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_readme(
    output_dir: Path,
    metrics: pd.DataFrame,
    selected_classifier: str,
    demo_stats: dict,
) -> None:
    overall = metrics[
        (metrics["scope"] == "out_of_fold_overall")
        & (metrics["task"] == "4class")
    ].set_index("classifier")
    text = f"""# PPG 睡眠分期与自适应闹钟基准

## 运行

```powershell
python run_ppg_end2end_benchmark.py
```

脚本只下载 PhysioNet `sleep-accel` v1.0.0 的 `labels` 与 `heart_rate` 文本文件，不访问 `motion` 或 `steps`。

## 固定设定

- 30 秒因果 epoch；所有特征只使用该 epoch 结束时已经到达的 PPG 心率采样。
- 以受试者为 group 的 `GroupKFold(n_splits=5)`，同一人不会跨训练/测试折。
- 唤醒回测统一使用 `{selected_classifier}` 的 OOF 概率；两类分类器均保留在分期指标 CSV。
- Unknown epoch 不参与分类指标，也不会触发提前唤醒；窗口到期始终强制兜底。
- 动态阈值按每位受试者的 OOF Wake+Light 得分分布标定：85% 分位数为上阈值，25% 分位数为下阈值。

## 主要结果

Logistic Regression 四分类 OOF Macro-F1：{overall.loc["LogisticRegression", "macro_f1"]:.4f}。

HistGradientBoosting 四分类 OOF Macro-F1：{overall.loc["HistGradientBoosting", "macro_f1"]:.4f}。

图 B 的代表性试次：受试者 {demo_stats.get("sid", "未知")}，W={demo_stats.get("window_minutes", 20)} 分钟，真实触发分期为 {STAGE_NAMES[int(demo_stats.get("true_stage", 0))]}。

## 输出文件

- `ppg_staging_metrics.csv`
- `ppg_oof_predictions.csv`
- `ppg_alarm_end2end_results.csv`
- `ppg_alarm_summary.csv`
- `real_ppg_end2end_benchmark.png`
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        download_data(args.data_dir, args.skip_download, args.force_download)
        subjects = load_subjects(args.data_dir)
        print("提取因果 PPG/PRV 特征……", flush=True)
        feature_parts = []
        for number, subject in enumerate(subjects, start=1):
            part = extract_causal_features(subject)
            feature_parts.append(part)
            print(
                f"  [{number:02d}/{len(subjects)}] {subject.sid}: "
                f"{len(part)} epochs",
                flush=True,
            )
        features = pd.concat(feature_parts, ignore_index=True)
        features_path = args.data_dir / "causal_ppg_features.csv"
        features.to_csv(features_path, index=False)
        print(
            f"特征完成：{len(features)} epochs，{features['sid'].nunique()} 名受试者，"
            f"Unknown {features['is_unknown'].mean() * 100:.1f}%。",
            flush=True,
        )

        metrics, oof, selected_classifier = train_subject_wise_cv(
            features, args.output_dir
        )
        print(f"唤醒回测统一分类器：{selected_classifier}", flush=True)
        alarm_results, summary, demo_stats = run_alarm_benchmark(
            oof, selected_classifier, args.output_dir
        )
        print(
            f"回测完成：{len(alarm_results)} 个策略-窗口-受试者试次。",
            flush=True,
        )

        raw_subjects = {subject.sid: subject for subject in subjects}
        plot_benchmark(
            features,
            oof,
            selected_classifier,
            alarm_results,
            summary,
            demo_stats,
            raw_subjects,
            args.output_dir / "real_ppg_end2end_benchmark.png",
            args.demo_subject,
        )
        write_readme(args.output_dir, metrics, selected_classifier, demo_stats)
        print(f"输出目录：{args.output_dir}", flush=True)
        print(summary.to_string(index=False), flush=True)
        return 0
    except Exception as exc:
        print(f"运行失败：{exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
