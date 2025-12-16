import numpy as np
import numpy.typing as npt
import helia_edge as helia

from ..defines import HKTaskParams
from .backend import HKInferenceBackend


class PcBackend(HKInferenceBackend):
    def __init__(self, params: HKTaskParams) -> None:
        """PC inference engine backend.

        This backend runs inference on a PC using Keras or TFLite models.

        Args:
            params (HKTaskParams): Task parameters
        """

        super().__init__(params=params)
        self._inputs = None
        self._outputs = None
        self._model = None

    def _is_tf_model(self) -> bool:
        ext = self.params.model_file.suffix
        return ext in [".h5", ".hdf5", ".keras", ".tf"]

    def open(self):
        """This method will simply load the keras or TFLite model"""
        if self._is_tf_model():
            self._model = helia.models.load_model(self.params.model_file)
        else:
            with open(self.params.model_file, "rb") as fp:
                model_content = fp.read()
            self._model = helia.interpreters.tflite.TfLiteKerasInterpreter(model_content=model_content)

    def close(self):
        """This method will unload the model"""
        self._model = None

    def set_inputs(self, inputs: npt.NDArray):
        """Set inputs for inference

        Args:
            inputs (npt.NDArray): Inputs for inference
        """
        self._inputs = inputs

    def perform_inference(self):
        """Perform inference on the model"""
        if self._is_tf_model():
            self._outputs = self._model.predict(np.expand_dims(self._inputs, 0), verbose=0).squeeze(0)
        else:
            self._outputs = self._model.predict(self._inputs)

    def get_outputs(self) -> npt.NDArray:
        """Get outputs from inference

        Returns:
            npt.NDArray: Outputs from inference
        """
        return self._outputs
