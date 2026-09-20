"""Проверки ошибок, которые способны незаметно завысить качество или испортить CSV.

Запуск: python -m unittest -v test_solution
"""

import unittest
import numpy as np
import pandas as pd

from features import prepare_events, build_features
from metric import precision_at_recall
from training import check_submission, temporal_masks


class SolutionTests(unittest.TestCase):
    def setUp(self):
        self.start = pd.Timestamp("2026-04-10")
        self.meta = pd.DataFrame({
            "cookie_id": ["a", "empty"],
            "cookie_created_at": [self.start - pd.Timedelta(days=7)] * 2,
            "window_start_ts": [self.start] * 2,
            "window_end_ts": [self.start + pd.Timedelta(days=1)] * 2,
        })
        records = []
        for sec, name, eid, page in [(0, "search_results_view", 100, 1),
                                     (20, "search_results_view", 100, 2),
                                     (20, "search_results_view", 100, 3),
                                     (60, "item_view", 200, np.nan)]:
            records.append({"cookie_id": "a", "event_ts": self.start + pd.Timedelta(seconds=sec),
                "eid": eid, "event_name": name, "platform": "WEB", "user_agent": "Mozilla Chrome/120.0",
                "item_id": 123 if name == "item_view" else np.nan,
                "item_category": "phones", "item_location": "city", "seller_type": "private",
                "search_query": " phone  test " if eid == 100 else None,
                "search_page": page, "pointer_x": sec, "pointer_y": sec + page})
        self.events = pd.DataFrame(records)

    def features(self, events):
        clean, audit = prepare_events(events, self.meta)
        return build_features(clean, self.meta), clean, audit

    def test_future_and_left_boundary(self):
        expected, _, _ = self.features(self.events)
        outside = pd.concat([self.events.iloc[[0]]] * 3, ignore_index=True)
        outside["event_ts"] = [self.start - pd.Timedelta(seconds=1),
                               self.start + pd.Timedelta(days=1),
                               self.start + pd.Timedelta(days=2)]
        got, clean, audit = self.features(pd.concat([self.events, outside], ignore_index=True))
        pd.testing.assert_frame_equal(expected, got)
        self.assertEqual(len(clean), 4)  # Событие ровно на левой границе сохранено.
        self.assertEqual(audit["after_or_at_window_end"], 2)
        self.assertEqual(audit["before_window"], 1)

    def test_exact_duplicates_and_same_time(self):
        expected, _, _ = self.features(self.events)
        got, clean, audit = self.features(pd.concat([self.events, self.events.iloc[[0]]], ignore_index=True))
        pd.testing.assert_frame_equal(expected, got)
        self.assertEqual(audit["duplicates_in_window"], 1)
        self.assertEqual(len(clean), 4)  # Два разных события с одним временем не удаляем.

    def test_input_order_does_not_change_features(self):
        expected, _, _ = self.features(self.events)
        got, _, _ = self.features(self.events.sample(frac=1, random_state=13))
        pd.testing.assert_frame_equal(expected, got)

    def test_empty_cookie_and_one_event(self):
        x, _, _ = self.features(self.events.iloc[:1])
        self.assertEqual(x.loc["empty", "n_events"], 0)
        self.assertEqual(x.loc["empty", "no_events"], 1)
        self.assertEqual(x.loc["a", "gap_median"], -1)
        self.assertTrue(np.isfinite(x.to_numpy()).all())

    def test_metric_ties_and_nonmonotone_precision(self):
        self.assertEqual(precision_at_recall([1, 0, 1, 1, 1], [5, 4, 3, 2, 1]), .8)
        self.assertEqual(precision_at_recall([1, 1, 0, 0], [.5] * 4), .5)
        self.assertEqual(precision_at_recall([0, 0, 1, 1], [.5] * 4), .5)

    def test_time_split_has_no_overlap(self):
        meta = self.meta.iloc[[0]].copy()
        next_day = meta.copy()
        next_day["window_start_ts"] += pd.Timedelta(days=1)
        next_day["window_end_ts"] += pd.Timedelta(days=1)
        m = pd.concat([meta, next_day], ignore_index=True)
        fit, valid = temporal_masks(m, "2026-04-11", "2026-04-12")
        np.testing.assert_array_equal(fit, [True, False])
        np.testing.assert_array_equal(valid, [False, True])

    def test_submission_rejects_missing_duplicate_or_invalid_scores(self):
        test = pd.DataFrame({"cookie_id": ["a", "b"]})
        correct = pd.DataFrame({"cookie_id": ["a", "b"], "score": [.1, .9]})
        check_submission(correct, test)
        bad_cases = [correct.iloc[:1], correct.assign(cookie_id=["a", "a"]),
                     correct.assign(score=[np.nan, .9]), correct.assign(score=[.1, 1.1]),
                     correct.iloc[::-1], correct.assign(target=[0, 1])]
        for bad in bad_cases:
            with self.assertRaises(ValueError):
                check_submission(bad, test)


if __name__ == "__main__":
    unittest.main()
