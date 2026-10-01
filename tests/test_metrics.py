from deeptrace.evaluation import metrics as mt


def test_known_values():
    m = mt.classification_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.5)
    assert m["confusion"] == {"tp": 1, "fp": 0, "fn": 1, "tn": 2}
    assert m["accuracy"] == 0.75 and m["balanced_accuracy"] == 0.75
    assert m["precision"] == 1.0 and m["recall"] == 0.5 and m["specificity"] == 1.0
    assert abs(m["f1"] - 2 / 3) < 1e-12
    assert m["roc_auc"] == 0.75


def test_degenerate_cases_give_none_not_errors():
    assert mt.classification_metrics([1, 1, 1], [0.9, 0.8, 0.7])["roc_auc"] is None
    m = mt.classification_metrics([0, 1], [0.1, 0.2])        # nothing predicted MANIPULATED
    assert m["precision"] is None and m["f1"] is None and m["recall"] == 0.0


def test_per_method_report():
    recs = [{"image_id": "real/a", "fake_prob": 0.2}, {"image_id": "real/b", "fake_prob": 0.7},
            {"image_id": "insight/a", "fake_prob": 0.9}, {"image_id": "insight/b", "fake_prob": 0.8}]
    r = mt.per_method_report(recs)
    assert r["real"] == {"n": 2, "share_predicted_manipulated": 0.5}
    assert r["insight"]["share_predicted_manipulated"] == 1.0
