import numpy as np

from analyze_e6_baseline_spectrum_predictor import ridge_fit_predict


def test_ridge_training_standardization_is_applied_to_test_values():
    x_train = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.asarray([0.0, 1.0, 2.0, 3.0])
    predictions, coefficients, mean, scale, intercept = ridge_fit_predict(
        x_train, y_train, np.asarray([[mean_value] for mean_value in (1.5, 2.5)])
    )
    assert mean[0] == 1.5
    assert scale[0] > 0
    assert intercept == 1.5
    assert coefficients[0] > 0
    assert predictions[1] > predictions[0]
