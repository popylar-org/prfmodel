"""CF stimulus encoding classes."""

import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.models.base import BaseEncoder
from prfmodel.stimuli import CFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import get_dtype


class CFStimulusEncoder(BaseEncoder[CFStimulus]):
    """
    Encoding model for connective field stimuli.

    Multiplies a source response with a connective model response and sums over the vertices in the source response.

    """

    @property
    def parameter_names(self) -> list:
        """Does not have any parameters. Returns an empty list."""
        return []

    @doc
    def __call__(
        self,
        stimulus: CFStimulus,
        response: Tensor,
        parameters: pd.DataFrame,  # noqa: ARG002 (unused method argument)
        dtype: str | None = None,
    ) -> Tensor:
        """Encode a Connective field model response with a source response.

        Parameters
        ----------
        %(stimulus_cf)s
        response : Tensor
            Connective field response.
        %(parameters)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = get_dtype(dtype)
        response = ops.convert_to_tensor(response, dtype=dtype)
        source_response = ops.convert_to_tensor(stimulus.source_response, dtype=dtype)

        if response.shape[1] != source_response.shape[0]:
            msg = (
                f"Second dimension of connective field response {response.shape[1]} does not match first dimension "
                f"of source response {source_response.shape[0]}"
            )
            raise ValueError(msg)

        return ops.tensordot(response, source_response, axes=[[1], [0]])
