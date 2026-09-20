"""Признаки поведения за одно окно. Здесь нет обучения и доступа к target."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


EVENT_NAMES = (
    "search_results_view", "item_view", "photo_swipe", "seller_page_view",
    "contact_phone_show", "contact_chat_open", "contact_message_sent",
    "favorite_add", "login", "captcha_shown",
)
META_COLUMNS = ["cookie_id", "cookie_created_at", "window_start_ts", "window_end_ts"]
PLATFORMS = ("web", "android", "ios", "unknown")
UA_FAMILIES = ("http_client", "headless", "avito_app", "yandex", "edge",
               "firefox", "chrome", "safari", "other")


def load_data(data_dir: str | Path):
    """Читаем cookie_id как строку, даже если формат ID в следующей выгрузке изменится."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv", dtype={"cookie_id": str})
    test = pd.read_csv(data_dir / "test.csv", dtype={"cookie_id": str})
    events = pd.read_csv(data_dir / "events.csv.gz", dtype={"cookie_id": str})
    for frame in (train, test):
        for col in META_COLUMNS[1:]:
            frame[col] = pd.to_datetime(frame[col], errors="raise", format="mixed")
        if frame[META_COLUMNS].isna().any().any() or frame.cookie_id.duplicated().any():
            raise ValueError("В метаданных есть пропуски или повторяющиеся cookie_id")
        if not frame.window_end_ts.gt(frame.window_start_ts).all():
            raise ValueError("Неверные границы окна")
        if frame.cookie_created_at.gt(frame.window_end_ts).any():
            raise ValueError("Дата создания cookie находится в будущем")
    if set(train.cookie_id) & set(test.cookie_id):
        raise ValueError("Train и test пересекаются по cookie_id")
    if not train.target.isin([0, 1]).all() or train.target.nunique() != 2:
        raise ValueError("Ожидается бинарная разметка 0/1 с обоими классами")
    events["event_ts"] = pd.to_datetime(events.event_ts, errors="raise", format="mixed")
    if events[["cookie_id", "event_ts"]].isna().any().any():
        raise ValueError("У события отсутствует cookie_id или время")
    return train, test, events


def prepare_events(events: pd.DataFrame, meta: pd.DataFrame):
    """Сначала ограничиваем окно, затем удаляем полные дубликаты сырых событий.

    Правая граница не включается: событие ровно в window_end_ts уже относится
    к следующему дню. Число исключённых событий остаётся только в аудите.
    """
    if meta.cookie_id.duplicated().any():
        raise ValueError("Ожидается одно окно на cookie_id")
    joined = events.merge(meta[META_COLUMNS], on="cookie_id", how="left",
                          validate="many_to_one", indicator=True)
    known = joined["_merge"].eq("both")
    inside = known & joined.event_ts.ge(joined.window_start_ts) & joined.event_ts.lt(joined.window_end_ts)
    ev = joined.loc[inside, events.columns].copy()
    n_duplicates = int(ev.duplicated().sum())
    ev = ev.drop_duplicates().reset_index(drop=True)
    audit = {
        "raw_events": len(events),
        "unknown_cookie_events": int((~known).sum()),
        "before_window": int((known & joined.event_ts.lt(joined.window_start_ts)).sum()),
        "after_or_at_window_end": int((known & joined.event_ts.ge(joined.window_end_ts)).sum()),
        "in_window_before_dedup": int(inside.sum()),
        "duplicates_in_window": n_duplicates,
        "clean_events": len(ev),
        "cookies_without_events": int((~meta.cookie_id.isin(ev.cookie_id)).sum()),
    }
    # Нормализация не должна превращать два разных сырых события в одно.
    p = ev.platform.fillna("").str.strip().str.lower()
    ev["platform_norm"] = p.replace({"desktop": "web", "iphone": "ios"})
    ev.loc[~ev.platform_norm.isin(PLATFORMS), "platform_norm"] = "unknown"
    for col in ["event_name", "item_category", "item_location", "seller_type", "search_query"]:
        ev[col] = ev[col].astype("string").str.strip().str.lower().replace("", pd.NA)
    ev["search_query"] = ev.search_query.str.replace(r"\s+", " ", regex=True)
    # Парсим каждый уникальный UA один раз. Версия браузера и модель телефона
    # не попадают в признаки: они легко меняются при обновлении клиента.
    ua_map = {ua: parse_user_agent(ua) for ua in ev.user_agent.fillna("").unique()}
    ev["ua_family"] = ev.user_agent.fillna("").map(lambda s: ua_map[s][0])
    ev["ua_os"] = ev.user_agent.fillna("").map(lambda s: ua_map[s][1])
    ev = ev.sort_values(["cookie_id", "event_ts", "eid", "item_id", "search_query"],
                        kind="mergesort", na_position="last").reset_index(drop=True)
    return ev, audit


def parse_user_agent(value: str) -> tuple[str, str]:
    s = str(value).lower()
    if re.search(r"requests|urllib|curl|scrapy|go-http-client|node-fetch|aiohttp|httpx", s):
        family = "http_client"
    elif "headless" in s:
        family = "headless"
    elif s.startswith("avito/"):
        family = "avito_app"
    elif "yabrowser" in s:
        family = "yandex"
    elif "edg/" in s:
        family = "edge"
    elif "firefox" in s:
        family = "firefox"
    elif "chrome" in s or "chromium" in s:
        family = "chrome"
    elif "safari" in s:
        family = "safari"
    else:
        family = "other"
    if "android" in s:
        os_name = "android"
    elif "iphone" in s or "ipad" in s:
        os_name = "ios"
    elif "windows" in s:
        os_name = "windows"
    elif "macintosh" in s:
        os_name = "mac"
    elif "linux" in s:
        os_name = "linux"
    else:
        os_name = "unknown"
    return family, os_name


def _stats(values, prefix: str) -> dict:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    names = ("mean", "std", "min", "p10", "median", "p90", "max", "cv", "iqr_ratio")
    if not len(a):
        return {f"{prefix}_{name}": np.nan for name in names}
    q10, q25, med, q75, q90 = np.quantile(a, [.1, .25, .5, .75, .9])
    mean, std = a.mean(), a.std()
    vals = (mean, std, a.min(), q10, med, q90, a.max(),
            std / (mean + 1e-6), (q75 - q25) / (med + 1e-6))
    return dict(zip((f"{prefix}_{name}" for name in names), vals))


def _diversity(values, prefix: str) -> dict:
    counts = pd.Series(values).dropna().value_counts().to_numpy(dtype=float)
    n = counts.sum()
    if not n:
        return {f"{prefix}_nunique": 0, f"{prefix}_unique_ratio": np.nan,
                f"{prefix}_top_share": np.nan, f"{prefix}_entropy": np.nan}
    p = counts / n
    return {f"{prefix}_nunique": len(counts), f"{prefix}_unique_ratio": len(counts) / n,
            f"{prefix}_top_share": p.max(), f"{prefix}_entropy": float(-(p * np.log(p)).sum())}


def _peak_count(seconds: np.ndarray, width: float) -> int:
    if not len(seconds):
        return 0
    # Максимум в скользящем полуоткрытом интервале [t, t + width).
    return int((np.searchsorted(seconds, seconds + width, side="left") - np.arange(len(seconds))).max())


def _cookie_features(g: pd.DataFrame, row) -> dict:
    n = len(g)
    age_days = (row.window_end_ts - row.cookie_created_at).total_seconds() / 86400
    f = {"n_events": n, "no_events": int(n == 0), "cookie_age_log_days": np.log1p(max(age_days, 0))}
    names = g.event_name.fillna("unknown").to_numpy(dtype=str)
    event_counts = pd.Series(names).value_counts()
    for name in EVENT_NAMES:
        f[f"count_{name}"] = int(event_counts.get(name, 0))
        f[f"share_{name}"] = float(event_counts.get(name, 0) / max(n, 1))
    f["event_nunique"] = len(event_counts)
    f.update(_diversity(names, "event"))
    for col, prefix in [("item_id", "item"), ("item_category", "category"),
                        ("item_location", "location"), ("search_query", "query")]:
        f.update(_diversity(g[col], prefix))
        f[f"{prefix}_missing_share"] = float(g[col].isna().mean()) if n else np.nan
    for col, values, prefix in [("platform_norm", PLATFORMS, "platform"),
                                 ("ua_family", UA_FAMILIES, "ua"),
                                 ("ua_os", ("windows", "mac", "linux", "android", "ios", "unknown"), "os")]:
        counts = g[col].value_counts()
        f[f"{prefix}_nunique"] = len(counts)
        for value in values:
            f[f"{prefix}_{value}_share"] = float(counts.get(value, 0) / max(n, 1))
    f["seller_pro_share"] = float(g.seller_type.eq("pro").fillna(False).sum() / max(g.seller_type.notna().sum(), 1))
    views = f["count_item_view"]
    searches = f["count_search_results_view"]
    contacts = sum(f[f"count_{s}"] for s in ("contact_phone_show", "contact_chat_open", "contact_message_sent"))
    f["contacts_per_view"] = contacts / max(views, 1)
    f["photos_per_view"] = f["count_photo_swipe"] / max(views, 1)
    f["favorites_per_view"] = f["count_favorite_add"] / max(views, 1)
    f["views_per_search"] = views / max(searches, 1)
    f["phone_per_chat"] = f["count_contact_phone_show"] / (1 + f["count_contact_chat_open"])
    f.update(_diversity(g.loc[g.event_name.eq("item_view"), "item_id"], "view_item"))

    seconds = (g.event_ts - row.window_start_ts).dt.total_seconds().to_numpy()
    dt = np.diff(seconds)
    f.update(_stats(dt, "gap"))
    # Длинные перерывы отделяют сессии. Регулярность внутри сессии считаем отдельно.
    short = dt[(dt > 0) & (dt <= 300)]
    f.update(_stats(short, "short_gap"))
    f["gap_zero_share"] = float(np.mean(dt == 0)) if len(dt) else np.nan
    for limit in (1, 5, 15, 60):
        f[f"gap_le_{limit}s_share"] = float(np.mean(dt <= limit)) if len(dt) else np.nan
    f["gap_over_30m_share"] = float(np.mean(dt > 1800)) if len(dt) else np.nan
    f["gap_unique_ratio"] = len(np.unique(dt)) / len(dt) if len(dt) else np.nan
    f["gap_adjacent_change_median"] = float(np.median(np.abs(np.diff(dt)))) if len(dt) > 1 else np.nan
    if len(short):
        _, counts = np.unique(short, return_counts=True)
        f["short_gap_mode_share"] = counts.max() / len(short)
        f["short_gap_burstiness"] = (short.std() - short.mean()) / (short.std() + short.mean() + 1e-6)
    else:
        f["short_gap_mode_share"] = f["short_gap_burstiness"] = np.nan
    f["active_span_seconds"] = seconds[-1] - seconds[0] if n else 0
    f["first_event_offset"] = seconds[0] if n else np.nan
    f["last_event_to_end"] = (row.window_end_ts - g.event_ts.iloc[-1]).total_seconds() if n else np.nan
    f["events_per_active_minute"] = 60 * n / max(f["active_span_seconds"], 60)
    for width in (10, 60, 300):
        f[f"peak_{width}s_count"] = _peak_count(seconds, width)
        f[f"peak_{width}s_share"] = f[f"peak_{width}s_count"] / max(n, 1)
    hours = (seconds // 3600).astype(int)
    f.update(_diversity(hours, "hour"))
    f["active_minutes"] = len(np.unique(seconds // 60))
    f["night_share"] = float(np.mean(hours < 6)) if n else np.nan
    for begin in (0, 6, 12, 18):
        f[f"hour_{begin}_{begin + 6}_share"] = float(np.mean((hours >= begin) & (hours < begin + 6))) if n else np.nan
    if n:
        starts = np.r_[0, np.flatnonzero(dt > 1800) + 1]
        ends = np.r_[starts[1:], n]
        session_sizes = ends - starts
        durations = seconds[ends - 1] - seconds[starts]
        f["session_count"] = len(starts)
        f["session_max_events"] = int(session_sizes.max())
        f["session_single_event_share"] = float(np.mean(session_sizes == 1))
        f["session_total_seconds"] = durations.sum()
        f["session_max_seconds"] = durations.max()
        f["session_mean_events"] = session_sizes.mean()
    else:
        for name in ("count", "max_events", "single_event_share", "total_seconds", "max_seconds", "mean_events"):
            f[f"session_{name}"] = 0

    # При совпадении timestamp порядок действий неизвестен. Переходы с такой
    # неоднозначностью не используем, вместо этого оставляем gap_zero_share.
    unique_time = ~g.event_ts.duplicated(keep=False).to_numpy()
    usable = unique_time[1:] & unique_time[:-1] & (dt > 0) & (dt <= 1800)
    pairs = list(zip(names[:-1][usable], names[1:][usable]))
    f["same_event_transition_share"] = float(np.mean([a == b for a, b in pairs])) if pairs else np.nan
    f["transition_unique_ratio"] = len(set(pairs)) / len(pairs) if pairs else np.nan
    for a, b, tag in [("search_results_view", "item_view", "search_to_view"),
                       ("item_view", "photo_swipe", "view_to_photo"),
                       ("item_view", "contact_phone_show", "view_to_phone")]:
        f[f"transition_{tag}_share"] = sum(x == a and y == b for x, y in pairs) / max(len(pairs), 1)

    search = g.loc[g.event_name.eq("search_results_view")]
    f.update(_stats(search.search_page.to_numpy(), "search_page"))
    pages = search.search_page.dropna().to_numpy()
    f["search_page_nunique"] = len(np.unique(pages))
    f["search_page_deep_share"] = float(np.mean(pages >= 5)) if len(pages) else np.nan
    page_diff = search.search_page.diff().to_numpy()[1:]
    query_same = (search.search_query.eq(search.search_query.shift()).fillna(False)).to_numpy()[1:]
    query_dt = search.event_ts.diff().dt.total_seconds().to_numpy()[1:]
    search_unique_time = ~search.event_ts.duplicated(keep=False).to_numpy()
    good = (query_same & (query_dt > 0) & (query_dt <= 1800) & np.isfinite(page_diff)
            & search_unique_time[1:] & search_unique_time[:-1])
    steps = page_diff[good]
    f["search_next_page_share"] = float(np.mean(steps == 1)) if len(steps) else np.nan
    f["search_repeated_page_share"] = float(np.mean(steps == 0)) if len(steps) else np.nan
    queries = search.search_query.dropna()
    f["query_mean_chars"] = float(queries.str.len().mean()) if len(queries) else np.nan
    f["query_mean_words"] = float(queries.str.split().str.len().mean()) if len(queries) else np.nan

    pointer = g[["pointer_x", "pointer_y"]].dropna()
    f["pointer_observed_count"] = len(pointer)
    f["pointer_observed_share"] = len(pointer) / max(n, 1)
    f["pointer_unique_ratio"] = len(pointer.drop_duplicates()) / len(pointer) if len(pointer) else np.nan
    ordered_pointer = g.loc[unique_time, ["pointer_x", "pointer_y"]].dropna()
    if len(ordered_pointer) > 1:
        delta = np.diff(ordered_pointer.to_numpy(), axis=0)
        distance = np.sqrt((delta ** 2).sum(axis=1))
        # Абсолютные координаты зависят от экрана. Используем долю повторов
        # и относительную изменчивость расстояний между наблюдаемыми точками.
        f["pointer_same_point_share"] = float(np.mean(distance == 0))
        f["pointer_distance_cv"] = float(distance.std() / (distance.mean() + 1e-6))
        f["pointer_x_unique_ratio"] = pointer.pointer_x.nunique() / len(pointer)
        f["pointer_y_unique_ratio"] = pointer.pointer_y.nunique() / len(pointer)
    else:
        for name in ("same_point_share", "distance_cv", "x_unique_ratio", "y_unique_ratio"):
            f[f"pointer_{name}"] = np.nan
    f.update(_extra_timing_features(g, seconds, names))
    return f


def _extra_timing_features(g: pd.DataFrame, seconds: np.ndarray, names: np.ndarray) -> dict:
    """Проверяем, добавляют ли что-то устойчивые оценки ритма и интервалы по типам."""
    f = {}
    dt = np.diff(seconds)
    active = dt[(dt > 0) & (dt <= 1800)]
    if len(active):
        log_gap = np.log1p(active)
        med = np.median(active)
        f["extra_gap_log_std"] = log_gap.std()
        f["extra_gap_log_mean"] = log_gap.mean()
        f["extra_gap_mad_ratio"] = np.median(np.abs(active - med)) / (med + 1e-6)
        f["extra_gap_trimmed_cv"] = active[active <= np.quantile(active, .9)].std() / (med + 1e-6)
        f["extra_gap_p90_p10"] = np.quantile(active, .9) / (np.quantile(active, .1) + 1e-6)
        f["extra_gap_geomean_arithmean"] = np.exp(log_gap.mean()) / (active.mean() + 1)
    else:
        for key in ("log_std", "log_mean", "mad_ratio", "trimmed_cv", "p90_p10", "geomean_arithmean"):
            f[f"extra_gap_{key}"] = np.nan
    for event, label in [("item_view", "view"), ("search_results_view", "search")]:
        selected = seconds[names == event]
        gaps = np.diff(selected)
        within_session = gaps[(gaps > 0) & (gaps <= 1800)]
        f.update(_stats(within_session, f"extra_{label}_gap"))
    if len(dt) > 1:
        good = (dt[1:] > 0) & (dt[:-1] > 0) & (dt[1:] <= 1800) & (dt[:-1] <= 1800)
        diffs = np.abs(np.log1p(dt[1:][good]) - np.log1p(dt[:-1][good]))
    else:
        diffs = np.array([])
    f["extra_gap_log_change_mean"] = float(diffs.mean()) if len(diffs) else np.nan
    f["extra_gap_log_change_std"] = float(diffs.std()) if len(diffs) else np.nan
    # Связь разных действий с одним объявлением: после просмотра человек часто
    # листает фото или пишет продавцу. ID используется только для сопоставления.
    item_events = g.loc[g.item_id.notna()]
    if len(item_events):
        per_item = item_events.groupby("item_id").event_name.nunique()
        f["extra_item_multi_action_share"] = float((per_item > 1).mean())
        f["extra_item_actions_mean"] = float(per_item.mean())
    else:
        f["extra_item_multi_action_share"] = f["extra_item_actions_mean"] = np.nan
    viewed = set(g.loc[g.event_name.eq("item_view"), "item_id"].dropna())
    for event, label in [("photo_swipe", "photo"), ("contact_phone_show", "phone"), ("favorite_add", "favorite")]:
        ids = set(g.loc[g.event_name.eq(event), "item_id"].dropna())
        f[f"extra_{label}_items_viewed_share"] = len(ids & viewed) / len(ids) if ids else np.nan
    return f


def build_features(clean_events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Агрегаты независимы между cookie: ни test, ни соседние дни ничего не обучают."""
    rows = meta[META_COLUMNS]
    by_cookie = clean_events.groupby("cookie_id", sort=False).indices
    empty = clean_events.iloc[:0]
    records = []
    for row in rows.itertuples(index=False):
        indices = by_cookie.get(row.cookie_id)
        group = clean_events.iloc[indices] if indices is not None else empty
        records.append(_cookie_features(group, row))
    features = pd.DataFrame(records, index=pd.Index(rows.cookie_id, name="cookie_id"))
    # -1 означает «статистика не определена», например интервал у одного события.
    # Ноль для такой ситуации был бы настоящим сигналом одновременности событий.
    return features.replace([np.inf, -np.inf], np.nan).fillna(-1).astype(np.float64)


def feature_group(name: str) -> str:
    if name.startswith(("ua_", "os_", "platform_")):
        return "client"
    if name.startswith("pointer_"):
        return "pointer"
    if name.startswith(("extra_gap_", "extra_view_gap_", "extra_search_gap_")):
        return "timing"
    if name.startswith(("gap_", "short_gap_", "active_", "first_event_", "last_event_",
                        "events_per_", "peak_", "hour_", "night_", "session_")):
        return "timing"
    if name.startswith("cookie_age"):
        return "age"
    return "actions"
