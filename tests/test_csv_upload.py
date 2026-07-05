import unittest

import pandas as pd

from csv_upload import build_batch_predictions, extract_text_rows, find_text_column


class FakeClassifier:
    def __init__(self):
        self.seen_texts = []

    def __call__(self, texts):
        self.seen_texts = texts
        return [
            {"label": "POSITIVE" if "great" in text.lower() else "NEGATIVE", "score": 0.91}
            for text in texts
        ]


class CsvUploadTests(unittest.TestCase):
    def test_finds_xquik_tweet_text_column(self):
        dataframe = pd.DataFrame(
            {
                "Tweet Created At": ["2026-07-05T00:00:00Z"],
                "Tweet Text": ["Great product feedback"],
            }
        )

        self.assertEqual(find_text_column(dataframe), "Tweet Text")

    def test_extracts_non_empty_rows(self):
        dataframe = pd.DataFrame({"text": ["Great app", "", None, "Slow queue"]})

        text_column, text_rows = extract_text_rows(dataframe)

        self.assertEqual(text_column, "text")
        self.assertEqual(text_rows.tolist(), ["Great app", "Slow queue"])

    def test_builds_batch_prediction_dataframe(self):
        dataframe = pd.DataFrame({"Tweet Text": ["Great app", "Slow queue"]})
        classifier = FakeClassifier()

        results = build_batch_predictions(dataframe, classifier)

        self.assertEqual(classifier.seen_texts, ["Great app", "Slow queue"])
        self.assertEqual(results["predicted_sentiment"].tolist(), ["POSITIVE", "NEGATIVE"])
        self.assertEqual(results["confidence"].tolist(), [0.91, 0.91])


if __name__ == "__main__":
    unittest.main()
