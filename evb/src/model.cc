//*****************************************************************************
//*** TensorFlow
//#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"
#include "model_buffer.h"

//*****************************************************************************
//*** Tensorflow Globals
tflite::ErrorReporter *errorReporter = nullptr;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *modelInput = nullptr;
TfLiteTensor *modelOutput = nullptr;
constexpr int kTensorArenaSize = 1024 * 300;
alignas(16) uint8_t tensorArena[kTensorArenaSize];

void init_model() {
    /**
     * @brief Initialize TFLM model block
     *
     */
    static tflite::MicroErrorReporter micro_error_reporter;
    errorReporter = &micro_error_reporter;

    tflite::InitializeTarget();

    // Map the model into a usable data structure.
    model = tflite::GetModel(g_afib_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(errorReporter,
            "Model provided is schema version %d not equal to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION
        );
        return;
    }

    // This pulls in all the operation implementations we need.
    // static tflite::MicroMutableOpResolver<1> resolver;
    static tflite::AllOpsResolver resolver;
    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensorArena, kTensorArenaSize, errorReporter
    );
    interpreter = &static_interpreter;

    // Allocate memory from the tensorArena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(errorReporter, "AllocateTensors() failed");
        return;
    }

    size_t bytesUsed = interpreter->arena_used_bytes();
    if (bytesUsed > kTensorArenaSize) {
        TF_LITE_REPORT_ERROR(errorReporter,
            "Model requires %d bytes for arena but given only %d bytes.",
            bytesUsed, kTensorArenaSize
        );
    }

    // Obtain pointers to the model's input and output tensors.
    modelInput = interpreter->input(0);
    modelOutput = interpreter->output(0);
}


int model_inference(float32_t *x, uint32_t *y) {
    *y = 0;
    float32_t y_val = -99;
    for (int i = 0; i < modelInput->dims->data[1]; i++) {
        modelInput->data.f[i] = x[i];
    }
    TfLiteStatus invokeStatus = interpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
        return -1;
    }
    for (int i = 0; i < modelOutput->dims->data[1]; i++) {
        if (modelOutput->data.f[i] > y_val) {
            y_val = modelOutput->data.f[i];
            *y = i;
        }
        // ns_printf("y[%lu] = %0.2f\n", i, modelOutput->data.f[i]);
    }
    return 0;
}
