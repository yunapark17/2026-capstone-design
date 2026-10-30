"""노트북에 정의된 전처리 코드의 Fold 분리와 변수 매핑을 검증합니다."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / '수정용_캡스톤디자인.ipynb'
CELLS = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
NAMESPACE = {'pipeline_memory': None, 'class_ratio': 4.0}
for index in [3, 5]:
    source = '\n'.join(line for line in ''.join(CELLS[index]['source']).splitlines()
                       if not line.startswith('!'))
    tree = ast.parse(source)
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), 'exec'), NAMESPACE)
SecomPreprocessor = NAMESPACE['SecomPreprocessor']
CorrelationFilter = NAMESPACE['CorrelationFilter']
make_model = NAMESPACE['make_model']
rank_settings = NAMESPACE['rank_settings']


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({0: [1., np.nan, 3., 5.],
                                   1: [1., np.nan, np.nan, np.nan],
                                   2: [7., 7., 7., 7.]})

    def test_validation_data_cannot_change_columns_or_median(self):
        prep = SecomPreprocessor().fit(self.train)
        validation = pd.DataFrame({0: [np.nan, 1000.], 1: [8., 9.], 2: [8., 9.]})
        result = prep.transform(validation)
        self.assertEqual(result.columns.tolist(), ['sensor_0'])
        np.testing.assert_allclose(result.sensor_0, [3., 1000.])
        self.assertEqual(prep.numeric_columns_, [0])
        self.assertEqual(prep.imputer_.statistics_[0], 3.)

    def test_missing_flags_preserve_dropped_high_missing_column(self):
        prep = SecomPreprocessor(add_indicator=True).fit(self.train)
        result = prep.transform(self.train)
        self.assertEqual(result.columns.tolist(),
                         ['sensor_0', 'missing_sensor_0', 'missing_sensor_1'])
        np.testing.assert_array_equal(result.missing_sensor_1, [0, 1, 1, 1])
        np.testing.assert_array_equal(result.missing_sensor_0, [0, 1, 0, 0])
        self.assertFalse(result.isna().any().any())

    def test_input_column_order_is_validated(self):
        prep = SecomPreprocessor().fit(self.train)
        with self.assertRaisesRegex(ValueError, '열의 순서'):
            prep.transform(self.train[[2, 1, 0]])

    def test_constant_only_input_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, '비상수'):
            SecomPreprocessor().fit(pd.DataFrame({'a': [1, 1], 'b': [np.nan, np.nan]}))

    def test_correlation_filter_is_fitted_only_on_training_rows(self):
        train = pd.DataFrame({'a': [1., 2., 3., 4.], 'b': [2., 4., 6., 8.],
                              'c': [1., 4., 2., 3.]})
        filt = CorrelationFilter(.95).fit(train)
        self.assertEqual(filt.get_feature_names_out().tolist(), ['a', 'c'])
        validation = train.copy()
        validation['b'] = [4., 1., 8., 0.]
        self.assertEqual(filt.transform(validation).columns.tolist(), ['a', 'c'])

    def test_pipeline_feature_names_for_capped_and_all_features(self):
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=80, n_features=12, n_informative=5,
                                   weights=[.8, .2], random_state=42)
        X[0, 0] = np.nan
        for variant, top_n in [('missing_flags', 200), ('expanded_features', None),
                               ('correlation', 10)]:
            with self.subTest(variant=variant):
                model = make_model(top_n, 'RF_weight', variant)
                params = {'classifier__n_estimators': 6}
                if top_n is not None:
                    params['selector__estimator__n_estimators'] = 6
                model.set_params(**params)
                model.fit(pd.DataFrame(X), y)
                names = model.named_steps['preprocess'].get_feature_names_out()
                names = names[model.named_steps['variance'].get_support()]
                names = names[model.named_steps['correlation'].get_support()]
                selector = model.named_steps['selector']
                if not isinstance(selector, str):
                    names = names[selector.get_support()]
                self.assertEqual(len(names), model.named_steps['classifier'].n_features_in_)
                scores = model.predict_proba(pd.DataFrame(X[:5]))
                self.assertTrue(np.isfinite(scores).all())
                np.testing.assert_allclose(scores.sum(axis=1), 1.)

    def test_recall_target_and_precision_floor_are_enforced(self):
        rows = pd.DataFrame([
            dict(recall=.80, precision=.12, AP=.2),
            dict(recall=.70, precision=.15, AP=.3),
            dict(recall=.69, precision=.90, AP=.7),
            dict(recall=.99, precision=.099, AP=.4),
            dict(recall=.85, precision=.10, AP=.1),
        ])
        self.assertEqual(rank_settings(rows).index.tolist(), [1, 0, 4])

    def test_unmet_target_is_not_silently_relaxed(self):
        rows = pd.DataFrame([dict(recall=.65, precision=.50, AP=.3)])
        self.assertTrue(rank_settings(rows).empty)

    def test_invalid_selection_targets_are_rejected(self):
        rows = pd.DataFrame([dict(recall=.8, precision=.2, AP=.3)])
        for recall, precision in [(1.1, .1), (.7, -.1), (float('nan'), .1)]:
            with self.subTest(recall=recall, precision=precision), self.assertRaises(ValueError):
                rank_settings(rows, recall, precision)


if __name__ == '__main__':
    unittest.main()
