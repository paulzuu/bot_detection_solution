"""Пересоздать submission.csv из исходных CSV без запуска Jupyter."""

from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "4"

import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from features import load_data, prepare_events, build_features
from training import fit_predict, check_submission


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--verify", action="store_true", help="Сравнить SHA-256 с приложенным результатом")
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads((ROOT / "model_config.json").read_text(encoding="utf-8"))
    train, test, events = load_data(args.data_dir)
    meta = pd.concat([train.drop(columns="target"), test], ignore_index=True)
    clean, audit = prepare_events(events, meta)
    print(f'Оставлено {audit["clean_events"]:,} событий. Строим признаки...', flush=True)
    features = build_features(clean, meta)
    scores, _ = fit_predict(features.loc[train.cookie_id], train.target,
                             features.loc[test.cookie_id], config)
    submission = pd.DataFrame({"cookie_id": test.cookie_id.to_numpy(), "score": scores})
    check_submission(submission, test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False, float_format="%.12f", lineterminator="\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    if args.verify:
        expected = (ROOT / "reference_submission.sha256").read_text().strip().split()[0]
        if digest != expected:
            raise RuntimeError(f"SHA-256 не совпал: {digest}. Проверьте версии библиотек и конфигурацию.")
        print("Результат побайтово совпадает с приложенным submission.csv.")
    print(f"Создан {args.output.name}: {len(submission)} строк. SHA-256: {digest}")
    print(f"Время: {time.perf_counter() - started:.1f} с")


if __name__ == "__main__":
    main()
