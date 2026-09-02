"""Tests for the public facade / tensor-kernel split shared by every model class.

Model classes expose two entry points. :meth:`__call__` is the public facade: it accepts the user-facing
types (a :class:`pandas.DataFrame` of parameters, a :class:`~prfmodel.stimuli.Stimulus` of arrays),
validates them, converts them, and delegates. :meth:`call` is the tensor-only kernel the fitters wrap in a
backend compilation primitive.

The tests here pin the contract between the two: that they agree numerically, that validation happens on
the facade side, and that the tensor side is reachable without going through pandas at all.

"""

import keras
import numpy as np
import pandas as pd
import pytest
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse import ShiftedGammaImpulse
from prfmodel.impulse import TwoGammaImpulse
from prfmodel.impulse.base import BaseImpulse
from prfmodel.models.cf import GaussianCFModel
from prfmodel.models.cf import GaussianCFResponse
from prfmodel.models.prf import DelayedNormGaussian2DPRFModel
from prfmodel.models.prf import DivNormGaussian2DPRFModel
from prfmodel.models.prf import DoG2DPRFModel
from prfmodel.models.prf import Gaussian2DCSSPRFModel
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.models.prf import Gaussian2DPRFResponse
from prfmodel.models.prf import PRFStimulusEncoder
from prfmodel.protocols import ModelProtocol
from prfmodel.regressors import AdditiveRegressors
from prfmodel.scaling import Baseline
from prfmodel.scaling import BaselineAmplitude
from prfmodel.stimuli import CFStimulus
from prfmodel.stimuli import PRFStimulus
from prfmodel.utils import TensorFrame
from prfmodel.utils import as_tensor_frame
from tests.conftest import PRFStimulusSetup

DTYPE = "float32"


def _prf_params(num_units: int = 3) -> pd.DataFrame:
    """Parameters covering the pRF response, the derivative two-gamma impulse and baseline/amplitude."""
    return pd.DataFrame(
        {
            "mu_x": np.linspace(-1.0, 1.0, num_units),
            "mu_y": np.linspace(1.0, -1.0, num_units),
            "sigma": np.linspace(1.0, 2.0, num_units),
            "delay": np.full(num_units, 6.0),
            "dispersion": np.full(num_units, 0.9),
            "undershoot": np.full(num_units, 12.0),
            "u_dispersion": np.full(num_units, 0.9),
            "ratio": np.full(num_units, 0.48),
            "weight_deriv": np.full(num_units, 0.5),
            "baseline": np.full(num_units, 0.1),
            "amplitude": np.full(num_units, 1.2),
        },
    )


# Plausible values keyed by parameter name. Dual pRF models suffix their non-shared parameters
# (``sigma_center``, ``mu_x_surround``, ...), so the lookup falls back to the longest matching prefix.
_PARAMETER_VALUES = {
    "mu_x": 0.5,
    "mu_y": -0.5,
    "sigma": 1.5,
    "sigma_saturation": 1.0,
    "delay": 6.0,
    "dispersion": 0.9,
    "dispersion_normalization": 1.0,
    "undershoot": 12.0,
    "u_dispersion": 0.9,
    "ratio": 0.48,
    "weight_deriv": 0.5,
    "baseline": 0.1,
    "amplitude": 1.2,
    "shift": 0.0,
    "gain": 1.0,
    "n": 0.5,
    "center_index": 0.0,
}


def _value_for(name: str) -> float:
    """Look up a plausible value for a parameter, tolerating the suffixes dual models append."""
    if name in _PARAMETER_VALUES:
        return _PARAMETER_VALUES[name]

    candidates = [key for key in _PARAMETER_VALUES if name.startswith(f"{key}_")]

    if not candidates:
        msg = f"No test value defined for parameter {name!r}"
        raise KeyError(msg)

    return _PARAMETER_VALUES[max(candidates, key=len)]


# 'center_index' selects a row of the distance matrix, so it must stay a whole number in range.
_UNRAMPED_PARAMETERS = frozenset({"center_index", "shift"})


def _params_for(model: ModelProtocol, num_units: int = 3) -> pd.DataFrame:
    """Build a parameter frame covering exactly what a model declares it needs.

    Values differ between units, so that a prediction which accidentally broadcast one unit's parameters
    across all of them would not still match.

    """
    ramp = 1.0 + 0.1 * np.arange(num_units)

    return pd.DataFrame(
        {
            name: np.full(num_units, _value_for(name)) * (1.0 if name in _UNRAMPED_PARAMETERS else ramp)
            for name in model.parameter_names
        },
    )


class TestFacadeAndKernelAgree(PRFStimulusSetup):
    """Test that entering through the facade and entering through the kernel give the same prediction.

    This is the property the whole split rests on: the fitters call :meth:`call`, users and the docs call
    :meth:`__call__`, and a divergence between them would mean the thing being optimized is not the thing
    that was documented.

    """

    @pytest.mark.parametrize(
        "impulse_model",
        [DerivativeTwoGammaImpulse(), TwoGammaImpulse(), ShiftedGammaImpulse()],
        ids=lambda m: type(m).__name__,
    )
    def test_canonical_prf_model(self, stimulus: PRFStimulus, impulse_model: BaseImpulse):
        """Test agreement for the canonical pRF model across all three impulse models."""
        model = Gaussian2DPRFModel(impulse_model=impulse_model, scaling_model=BaselineAmplitude())
        params = _prf_params()

        if isinstance(impulse_model, ShiftedGammaImpulse):
            params["shift"] = 0.0

        facade = model(stimulus, params, dtype=DTYPE)
        kernel = model.call(stimulus.to_tensors(DTYPE), as_tensor_frame(params, DTYPE))

        np.testing.assert_allclose(keras.ops.convert_to_numpy(facade), keras.ops.convert_to_numpy(kernel))

    def test_response_submodel(self, stimulus: PRFStimulus):
        """Test agreement for a leaf response model, which has no submodels of its own."""
        model = Gaussian2DPRFResponse()
        params = _prf_params()[["mu_x", "mu_y", "sigma"]]

        facade = model(stimulus, params, dtype=DTYPE)
        kernel = model.call(stimulus.to_tensors(DTYPE), as_tensor_frame(params, DTYPE))

        np.testing.assert_allclose(keras.ops.convert_to_numpy(facade), keras.ops.convert_to_numpy(kernel))

    def test_stimulus_encoder_submodel(self, stimulus: PRFStimulus):
        """Test agreement for an encoder, whose kernel takes a response tensor as well as the stimulus."""
        encoder = PRFStimulusEncoder()
        params = _prf_params()
        response = Gaussian2DPRFResponse()(stimulus, params[["mu_x", "mu_y", "sigma"]], dtype=DTYPE)

        facade = encoder(stimulus, response, params, dtype=DTYPE)
        kernel = encoder.call(stimulus.to_tensors(DTYPE), response, as_tensor_frame(params, DTYPE))

        np.testing.assert_allclose(keras.ops.convert_to_numpy(facade), keras.ops.convert_to_numpy(kernel))

    def test_impulse_submodel(self):
        """Test agreement for an impulse model, whose kernel takes no stimulus at all."""
        impulse_model = DerivativeTwoGammaImpulse()
        params = _prf_params()

        facade = impulse_model(params, dtype=DTYPE)
        kernel = impulse_model.call(as_tensor_frame(params, DTYPE))

        np.testing.assert_allclose(keras.ops.convert_to_numpy(facade), keras.ops.convert_to_numpy(kernel))

    def test_scaling_submodel(self):
        """Test agreement for a scaling model, whose kernel takes a response tensor instead of a stimulus."""
        scaling_model = BaselineAmplitude()
        params = _prf_params()
        inputs = np.ones((3, 10))

        facade = scaling_model(inputs, params, dtype=DTYPE)
        kernel = scaling_model.call(keras.ops.convert_to_tensor(inputs, dtype=DTYPE), as_tensor_frame(params, DTYPE))

        np.testing.assert_allclose(keras.ops.convert_to_numpy(facade), keras.ops.convert_to_numpy(kernel))

    @pytest.mark.parametrize(
        "model_class",
        [DoG2DPRFModel, DivNormGaussian2DPRFModel, DelayedNormGaussian2DPRFModel, Gaussian2DCSSPRFModel],
        ids=lambda c: c.__name__,
    )
    def test_composite_prf_models(self, stimulus: PRFStimulus, model_class: type):
        """Test agreement for the composite pRF models.

        The dual models run their pRF submodel twice with suffixed parameter names, and the compressive and
        delayed-normalization models add stages after the encoding. Each reaches its submodels through
        `call`, so a submodel that still expected a data frame would fail here.

        """
        model = model_class()
        params = _params_for(model)

        facade = model(stimulus, params, dtype=DTYPE)
        kernel = model.call(stimulus.to_tensors(DTYPE), as_tensor_frame(params, DTYPE))

        np.testing.assert_allclose(keras.ops.convert_to_numpy(facade), keras.ops.convert_to_numpy(kernel))

    def test_connective_field_model(self):
        """Test agreement for the connective field family, which uses a different stimulus type."""
        rng = np.random.default_rng(0)
        num_vertices, num_frames = 8, 12
        stimulus = CFStimulus(
            distance_matrix=rng.uniform(0.0, 2.0, (num_vertices, num_vertices)),
            source_response=rng.normal(size=(num_vertices, num_frames)),
        )
        model = GaussianCFModel(scaling_model=BaselineAmplitude())
        params = pd.DataFrame(
            {"center_index": [0.0, 3.0], "sigma": [1.0, 1.5], "baseline": [0.1, 0.2], "amplitude": [1.0, 2.0]},
        )

        facade = model(stimulus, params, dtype=DTYPE)
        kernel = model.call(stimulus.to_tensors(DTYPE), as_tensor_frame(params, DTYPE))

        np.testing.assert_allclose(keras.ops.convert_to_numpy(facade), keras.ops.convert_to_numpy(kernel))


class TestValidationLivesOnTheFacade(PRFStimulusSetup):
    """Test that checks needing concrete Python values run in `__call__` and not in `call`.

    :meth:`call` is traced, so it cannot read a tensor back to a Python `bool`. Every check that has to do
    so therefore belongs to the facade. These tests pin where each check lives, because moving one the wrong
    way only fails once a backend compiles the model.

    """

    def test_missing_parameter_is_reported_by_the_facade(self, stimulus: PRFStimulus):
        """Test that a missing parameter column raises before any arithmetic happens."""
        model = Gaussian2DPRFResponse()

        with pytest.raises(ValueError, match="Missing required parameter names"):
            model(stimulus, pd.DataFrame({"mu_x": [0.0], "mu_y": [0.0]}))

    def test_out_of_domain_impulse_parameter_is_reported_by_the_facade(self):
        """Test that a non-positive impulse parameter raises eagerly rather than yielding a NaN response.

        The equivalent check inside :func:`~prfmodel.density.gamma_density` skips itself under a trace, so
        without this one a bad starting value would only show up as a NaN loss thousands of steps later.

        """
        params = _prf_params()
        params["dispersion"] = -1.0

        with pytest.raises(ValueError, match="'dispersion' must be > 0"):
            DerivativeTwoGammaImpulse()(params)

    def test_out_of_domain_impulse_parameter_is_reported_through_a_composite(self, stimulus: PRFStimulus):
        """Test that a composite model forwards the domain check to its impulse submodel.

        A composite reaches its submodels through their `call` methods, which skip their facades, so
        without an explicit fan-out this check would never run for the model a fitter actually holds --
        and a negative `delay` does not even produce a NaN to fall back on, only a finite wrong number.

        """
        params = _prf_params()
        params["delay"] = -1.0

        with pytest.raises(ValueError, match="'delay' must be > 0"):
            Gaussian2DPRFModel()(stimulus, params)

    def test_regressors_argument_is_validated_by_the_facade(self, stimulus: PRFStimulus):
        """Test that passing regressors to a model without a regressors submodel is rejected."""
        model = Gaussian2DPRFModel()

        with pytest.raises(ValueError, match="regressors_model"):
            model(stimulus, _prf_params(), regressors=pd.DataFrame({"reg": np.ones(50)}))


class TestExtraColumnsAreIgnored(PRFStimulusSetup):
    """Test the promise that a frame may carry columns no model reads.

    Stated in the :mod:`prfmodel.regressors` module docstring and repeated on every `regressors` parameter:
    "column order is unimportant and extra columns are silently ignored". A facade that converts a whole
    frame to tensors silently narrows that to "extra numeric columns", because a label column then has to
    survive `keras.ops.convert_to_tensor`. Each facade converts only the columns it reads instead.

    """

    def test_canonical_facade_ignores_a_label_column(self, stimulus: PRFStimulus):
        """A canonical model tolerates a non-numeric column among the parameters."""
        model = Gaussian2DPRFModel()
        params = _prf_params().assign(roi=["V1", "V2", "V3"])

        assert np.isfinite(np.asarray(model(stimulus, params))).all()

    def test_response_facade_ignores_a_label_column(self, stimulus: PRFStimulus):
        """A response model reads its own parameters only, so the rest need not be numeric."""
        response = Gaussian2DPRFResponse()
        params = _prf_params().assign(roi=["V1", "V2", "V3"])

        assert np.isfinite(np.asarray(response(stimulus, params))).all()

    def test_regressors_facade_ignores_an_extra_design_column(self):
        """A design column outside :attr:`names` need not be numeric."""
        regressors_model = AdditiveRegressors(names=["mx"])
        num_frames = 8
        design = pd.DataFrame({"mx": np.zeros(num_frames), "trial_type": ["rest"] * num_frames})

        resp = regressors_model(design, pd.DataFrame({"beta_mx": [1.0]}))

        assert np.asarray(resp).shape == (1, num_frames)

    def test_a_defaulted_impulse_parameter_can_still_be_overridden(self):
        """Test that narrowing the conversion does not drop an override of a defaulted parameter.

        A defaulted name is absent from :attr:`parameter_names` but the impulse response still reads it, so
        it has to survive the conversion. Filtering on the caller-facing names instead would silently ignore
        the override and return the default response.

        """
        impulse_model = TwoGammaImpulse()
        labels = {"roi": ["V1"]}

        default = np.asarray(impulse_model(pd.DataFrame({"undershoot": [12.0], **labels})))
        overridden = np.asarray(impulse_model(pd.DataFrame({"undershoot": [30.0], **labels})))

        assert not np.allclose(default, overridden)


class TestKernelTakesTensorsOnly(PRFStimulusSetup):
    """Test that the tensor side is reachable, and stays reachable, without pandas or numpy."""

    def test_kernel_runs_from_a_hand_built_tensor_frame(self, stimulus: PRFStimulus):
        """Test that a TensorFrame assembled from tensors is enough to drive a model.

        This is exactly what the fitters do: they hold parameters as backend variables and never
        materialize a data frame.

        """
        params = TensorFrame(
            {name: keras.ops.convert_to_tensor(col.to_numpy()) for name, col in _prf_params().items()},
            dtype=DTYPE,
        )

        prediction = Gaussian2DPRFModel().call(stimulus.to_tensors(DTYPE), params)

        assert keras.ops.shape(prediction) == (3, stimulus.design.shape[0])

    def test_to_tensors_carries_the_arrays_a_model_reads(self, stimulus: PRFStimulus):
        """Test that the tensor mirror holds the same values as the stimulus it came from."""
        tensors = stimulus.to_tensors(DTYPE)

        np.testing.assert_allclose(keras.ops.convert_to_numpy(tensors.design), stimulus.design, rtol=1e-6)
        np.testing.assert_allclose(keras.ops.convert_to_numpy(tensors.grid), stimulus.grid, rtol=1e-6)

    def test_to_tensors_returns_a_fresh_bundle_every_time(self, stimulus: PRFStimulus):
        """Test that the tensor bundle is not memoized on the stimulus.

        A tensor built inside a compiled function belongs to that function's graph and cannot be read from a
        later one. Caching the bundle would therefore leak the first trace into every subsequent one, which
        is the bug that `BaseImpulse.get_frames` had to be fixed for.

        """
        assert stimulus.to_tensors(DTYPE) is not stimulus.to_tensors(DTYPE)

    def test_response_model_kernel_never_touches_the_stimulus_object(self, stimulus: PRFStimulus):
        """Test that `call` reads only the tensor bundle, by passing it one built independently."""
        tensors = stimulus.to_tensors(DTYPE)
        params = as_tensor_frame(_prf_params()[["mu_x", "mu_y", "sigma"]], DTYPE)

        prediction = Gaussian2DPRFResponse().call(tensors, params)

        assert keras.ops.shape(prediction) == (3, *stimulus.grid.shape[:-1])


class TestSubclassesImplementTheKernel:
    """Test that the extension point moved from `__call__` to `call`."""

    @pytest.mark.parametrize(
        "model_class",
        [Gaussian2DPRFResponse, GaussianCFResponse, PRFStimulusEncoder, DerivativeTwoGammaImpulse, Baseline],
    )
    def test_concrete_models_define_call_and_inherit_the_facade(self, model_class: type):
        """Test that a model defines `call` itself and takes `__call__` from its base class.

        A subclass that overrode `__call__` would silently opt out of validation and conversion, and its
        `call` would never run.

        """
        assert "call" in vars(model_class), f"{model_class.__name__} does not implement 'call'"
        assert "__call__" not in vars(model_class), f"{model_class.__name__} overrides '__call__'"
