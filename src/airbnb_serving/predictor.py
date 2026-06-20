import pandas as pd

from airbnb_serving.schema import ListingFeatures, PredictionResponse


def _features_to_dict(features: ListingFeatures) -> dict:
    return features.model_dump()


def predict_single(features: ListingFeatures, model, run_id: str) -> PredictionResponse:
    row = _features_to_dict(features)
    X = pd.DataFrame([row])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return PredictionResponse(
        listing_id=None,
        prediction=int(prediction),
        probability_high_demand=float(probability),
        model_run_id=str(run_id),
    )


def predict_batch(
    features_list: list[ListingFeatures],
    model,
    run_id: str,
) -> list[PredictionResponse]:
    rows = [_features_to_dict(features) for features in features_list]
    X = pd.DataFrame(rows)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    responses = []
    for prediction, probability in zip(predictions, probabilities):
        responses.append(
            PredictionResponse(
                listing_id=None,
                prediction=int(prediction),
                probability_high_demand=float(probability),
                model_run_id=str(run_id),
            )
        )

    return responses
