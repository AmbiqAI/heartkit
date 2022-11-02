/**
 * @file model.cc
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Performs inference using TFLM
 * @version 0.1
 * @date 2022-11-02
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
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
constexpr int kTensorArenaSize = 1024 * 50;
alignas(16) uint8_t tensorArena[kTensorArenaSize];

int init_model() {
    /**
     * @brief Initialize TFLM model block
     *
     */
    static tflite::MicroMutableOpResolver<13> model_op_resolver;
    model_op_resolver.AddQuantize();
    model_op_resolver.AddShape();
    model_op_resolver.AddStridedSlice();
    model_op_resolver.AddPack();
    model_op_resolver.AddReshape();
    model_op_resolver.AddConv2D();
    model_op_resolver.AddMaxPool2D();
    model_op_resolver.AddAdd();
    model_op_resolver.AddMean();
    model_op_resolver.AddFullyConnected();
    model_op_resolver.AddSoftmax();
    model_op_resolver.AddRelu();
    model_op_resolver.AddDequantize();

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
        return 1;
    }

    // Build an TFLM interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, model_op_resolver, tensorArena, kTensorArenaSize, errorReporter
    );
    interpreter = &static_interpreter;

    // Allocate memory from the tensorArena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(errorReporter, "AllocateTensors() failed");
        return 1;
    }

    size_t bytesUsed = interpreter->arena_used_bytes();
    if (bytesUsed > kTensorArenaSize) {
        TF_LITE_REPORT_ERROR(errorReporter,
            "Model requires %d bytes for arena but given only %d bytes.",
            bytesUsed, kTensorArenaSize
        );
        return 1;
    }

    // Obtain pointers to the model's input and output tensors.
    modelInput = interpreter->input(0);
    modelOutput = interpreter->output(0);
    return 0;
}

int model_inference(float32_t *x, float32_t *y) {
    /**
     * @brief Run inference
     * @param x Model inputs
     * @param y Model outputs
     * @return Output label index
     */
    int y_idx = -1;
    float32_t y_val = -9999;
    for (int i = 0; i < modelInput->dims->data[1]; i++) {
        modelInput->data.f[i] = x[i];
    }
    TfLiteStatus invokeStatus = interpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
        return -1;
    }
    for (int i = 0; i < modelOutput->dims->data[1]; i++) {
        y[i] = modelOutput->data.f[i];
        if ((y_idx == -1) || (y[i] > y_val)) {
            y_val = y[i];
            y_idx = i;
        }
    }
    return y_idx;
}
