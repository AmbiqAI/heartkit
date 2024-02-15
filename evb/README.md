# HeartKit EVB

The included code configures the EVB to run as a TFLM inference engine that serves requests over USB using RPC. A client is able to connect and perform the following actions: send model as TFLM flatbuffer, send input data, perform inference, fetch output data. The EVB inference engine is used with each of the included task demos. By selecting "EVB" as the backend, the PC will connect and run inference on the EVB.

_NOTE_: The inference engine registers all available TFLM operations and allocates a large amount of memory for TFLM tensor arena and model flatbuffer to accomadate any model. This is not ideal for profiling power / memory. To get better power / memory analysis, please use neuralSPOT's AutoDeploy tool.
