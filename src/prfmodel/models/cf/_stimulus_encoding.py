"""Connective field stimulus encoding."""

from keras import ops
from prfmodel._docstring import doc
from prfmodel.models.base import BaseStimulusEncoder
from prfmodel.stimuli import CFStimulus
from prfmodel.stimuli import CFStimulusTensors
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame


class CFStimulusEncoder(BaseStimulusEncoder[CFStimulus, CFStimulusTensors]):
    """
    Encoding model for connective field stimuli.

    Multiplies a source response with a connective field model response and sums over the vertices in the source
    response.

    """

    @property
    def parameter_names(self) -> list:
        """Does not have any parameters. Returns an empty list."""
        return []

    @doc
    def call(self, stimulus: CFStimulusTensors, response: Tensor, parameters: TensorFrame) -> Tensor:  # noqa: ARG002 (this encoder has no parameters, but the signature is fixed by the base class)
        """Encode a connective field model response with a source response.

        Parameters
        ----------
        %(stimulus_cf_tensors)s
        response : Tensor
            Connective field response.
        %(parameters_tensors)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        source_response = stimulus.source_response

        if response.shape[1] != source_response.shape[0]:
            msg = (
                f"Second dimension of connective field response {response.shape[1]} does not match first dimension "
                f"of source response {source_response.shape[0]}"
            )
            raise ValueError(msg)

        return ops.tensordot(response, source_response, axes=[[1], [0]])
