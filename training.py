"""Обучение, временная валидация и проверка формата ответа."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from metric import precision_at_recall, recall_at_fpr
from features import feature_group

SEED = 42
THREADS = 4
# Два последовательных блока для выбора модели. Последние три дня пока не трогаем.
DEVELOPMENT_FOLDS = (("2026-04-13", "2026-04-15"), ("2026-04-15", "2026-04-17"))
HOLDOUT_START = "2026-04-17"


def temporal_masks(meta: pd.DataFrame, start: str, end: str | None = None):
    start = pd.Timestamp(start)
    train = meta.window_end_ts.le(start).to_numpy()
    valid = meta.window_start_ts.ge(start).to_numpy()
    if end is not None:
        valid &= meta.window_start_ts.lt(pd.Timestamp(end)).to_numpy()
    if not train.any() or not valid.any():
        raise ValueError("Пустой обучающий или проверочный блок")
    if meta.loc[train, "window_end_ts"].max() > meta.loc[valid, "window_start_ts"].min():
        raise ValueError("Окна обучения заходят в период проверки")
    return train, valid


def model_columns(X: pd.DataFrame, feature_set: str = "full") -> list[str]:
    if feature_set == "baseline":
        return ["n_events", "item_nunique"]
    if feature_set == "no_timing":
        return [c for c in X if feature_group(c) != "timing"]
    if feature_set == "no_client":
        return [c for c in X if feature_group(c) != "client"]
    if feature_set == "no_captcha_flags":
        return [c for c in X if "captcha" not in c]
    if feature_set == "original":
        return [c for c in X if not c.startswith("extra_")]
    if feature_set == "original_no_client":
        return [c for c in X if not c.startswith("extra_") and feature_group(c) != "client"]
    if feature_set != "full":
        raise ValueError(feature_set)
    return list(X.columns)


def make_model(kind: str = "catboost", depth: int = 6, iterations: int = 800,
               seed: int = SEED, positive_weight: float = 1):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                      random_state=seed, n_jobs=THREADS)
    if kind == "hist":
        # Встроенная случайная validation_fraction здесь не нужна:
        # проверяемся только по времени снаружи модели.
        return HistGradientBoostingClassifier(max_iter=350, learning_rate=.06,
                    max_leaf_nodes=15, l2_regularization=8, min_samples_leaf=30,
                    early_stopping=False, random_state=seed)
    if kind != "catboost":
        raise ValueError(kind)
    return CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=.04,
               l2_leaf_reg=8, loss_function="Logloss", random_seed=seed,
               class_weights=[1, positive_weight],
               thread_count=THREADS, task_type="CPU", verbose=False,
               allow_writing_files=False)


def evaluate(y, score) -> dict:
    return {
        "p_at_r70": precision_at_recall(y, score),
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "recall_at_fpr01": recall_at_fpr(y, score, .01),
        "n": len(y), "positives": int(np.sum(y)), "prevalence": float(np.mean(y)),
    }


def best_operating_point(y, score, min_recall: float = .70) -> dict:
    """Порог для разбора ошибок на данной размеченной выборке, не для сабмита.

    precision_recall_curve также объединяет одинаковые score. Последняя точка
    не имеет порога; убираем её перед выбором.
    """
    y, score = np.asarray(y), np.asarray(score)
    p, r, thresholds = precision_recall_curve(y, score)
    eligible = np.flatnonzero(r[:-1] >= min_recall)
    idx = eligible[np.argmax(p[:-1][eligible])]
    threshold = float(thresholds[idx])
    pred = score >= threshold
    return {"threshold": threshold, "precision": float(p[idx]), "recall": float(r[idx]),
            "tp": int(((y == 1) & pred).sum()), "fp": int(((y == 0) & pred).sum()),
            "fn": int(((y == 1) & ~pred).sum()), "tn": int(((y == 0) & ~pred).sum())}


def run_experiment(X, meta, config: dict, folds=DEVELOPMENT_FOLDS):
    cols = model_columns(X, config.get("feature_set", "full"))
    records, predictions = [], []
    for start, end in folds:
        fit_mask, val_mask = temporal_masks(meta, start, end)
        model = make_model(config.get("kind", "catboost"), config.get("depth", 6),
                           config.get("iterations", 800), config.get("seed", SEED),
                           config.get("positive_weight", 1))
        model.fit(X.loc[fit_mask, cols], meta.loc[fit_mask, "target"])
        score = model.predict_proba(X.loc[val_mask, cols])[:, 1]
        record = {"experiment": config["name"], "valid_start": start, "valid_end": end,
                  "n_features": len(cols), "train_n": int(fit_mask.sum()),
                  **evaluate(meta.loc[val_mask, "target"].to_numpy(), score)}
        records.append(record)
        predictions.append(pd.DataFrame({"cookie_id": meta.loc[val_mask, "cookie_id"].to_numpy(),
                "target": meta.loc[val_mask, "target"].to_numpy(), "score": score,
                "valid_start": start, "experiment": config["name"]}))
        print(f'{config["name"]:24} {start}: P@R70={record["p_at_r70"]:.4f}; AP={record["average_precision"]:.4f}', flush=True)
    return pd.DataFrame(records), pd.concat(predictions, ignore_index=True)


def bootstrap_metric(y, score, n_bootstrap: int = 1000, seed: int = SEED) -> dict:
    """Стратифицированный bootstrap cookie при уже зафиксированных предсказаниях.

    Интервал описывает шум этой выборки при фиксированной доле классов. Он не
    учитывает новые семейства ботов, корреляции источников и переобучение модели.
    """
    y, score = np.asarray(y), np.asarray(score)
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    values = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.r_[rng.choice(positive, len(positive), replace=True),
                    rng.choice(negative, len(negative), replace=True)]
        values[i] = precision_at_recall(y[idx], score[idx])
    return {"low": float(np.quantile(values, .025)), "high": float(np.quantile(values, .975)),
            "n_bootstrap": n_bootstrap}


def check_submission(submission: pd.DataFrame, test: pd.DataFrame):
    if list(submission.columns) != ["cookie_id", "score"]:
        raise ValueError("В submission нужны ровно две колонки: cookie_id, score")
    if len(submission) != len(test) or submission.cookie_id.duplicated().any():
        raise ValueError("Неправильное число строк или повторяющиеся cookie_id")
    if not np.array_equal(submission.cookie_id.to_numpy(), test.cookie_id.to_numpy()):
        raise ValueError("ID или порядок строк не совпадают с test.csv")
    if not np.isfinite(submission.score).all() or not submission.score.between(0, 1).all():
        raise ValueError("Оценки должны быть конечными числами от 0 до 1")


def fit_predict(X_train, y, X_test, config: dict):
    if "members" in config:
        parts = [fit_predict(X_train, y, X_test, member) for member in config["members"]]
        return np.mean([p[0] for p in parts], axis=0), [m for p in parts for m in p[1]]
    cols = model_columns(X_train, config.get("feature_set", "full"))
    scores, models = [], []
    for seed in config.get("seeds", [config.get("seed", SEED)]):
        model = make_model(config.get("kind", "catboost"), config.get("depth", 6),
                           config.get("iterations", 800), seed, config.get("positive_weight", 1))
        model.fit(X_train[cols], y)
        scores.append(model.predict_proba(X_test[cols])[:, 1])
        models.append(model)
    return np.mean(scores, axis=0), models
