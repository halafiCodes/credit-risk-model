import pandas as pd
from src.train_models import preprocess_data


class DummyPipeline:
    def transform(self, X):
        return X  # passthrough for testing


def test_preprocess_data():
    df = pd.DataFrame({
        "CustomerId": [1, 2],
        "is_high_risk": [0, 1],
        "feature1": [100, 200],
        "feature2": [1.5, 2.5]
    })

    pipeline = DummyPipeline()

    X, y = preprocess_data(df, pipeline)

    assert len(X) == 2
    assert len(y) == 2
