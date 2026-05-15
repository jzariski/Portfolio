from pathlib import Path

from ModelBuilder.ModelBuilder import ModelBuilder
from Parsing.TextParser import TextParser


ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / "ymd_example_small.txt").open("rb") as sample:
        data, lon, lat, headers = TextParser(sample).parseFile()

    builder = ModelBuilder(
        {
            "n_estimators": 25,
            "learning_rate": 0.03,
            "max_depth": 3,
            "early_stopping_rounds": 5,
        },
        data,
        "smoke-test",
        extraAugments=True,
        headers=headers,
    )
    builder.TrainTestSplit(split_by_Time=True, train_pct=0.65, eval_pct=0.85)
    builder.createModel()
    builder.evaluateModel()

    feature_values = {
        name: builder.feature_defaults[name]
        for name in builder.custom_feature_names
    }
    prediction = builder.makePrediction(
        mountRA=150.0,
        mountDec=30.0,
        second=26,
        minute=24,
        hour=22,
        day=7,
        month=6,
        year=2025,
        currList=[],
        extraOn=True,
        feature_values=feature_values,
    )

    print(f"Parsed {data.shape[0]} rows at lon={lon}, lat={lat}")
    print(f"Trained with {builder.X_train.shape[1]} input features: {builder.feature_names}")
    print(f"Prediction RA/Dec offset: {prediction[0][0]:.6f}, {prediction[0][1]:.6f}")


if __name__ == "__main__":
    main()
