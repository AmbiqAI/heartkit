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
#include "model.h"
#include "arrhythmia_model_buffer.h"
#include "beat_model_buffer.h"
#include "segmentation_model_buffer.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

//*****************************************************************************
//*** Tensorflow Globals
tflite::ErrorReporter *errorReporter = nullptr;

constexpr int arrTensorArenaSize = 1024 * 200;
alignas(16) static uint8_t arrTensorArena[arrTensorArenaSize];

constexpr int segTensorArenaSize = 1024 * 200;
alignas(16) static uint8_t segTensorArena[segTensorArenaSize];

constexpr int beatTensorArenaSize = 1024 * 100;
alignas(16) static uint8_t beatTensorArena[beatTensorArenaSize];

const tflite::Model *arrModel = nullptr;
tflite::MicroInterpreter *arrInterpreter = nullptr;
TfLiteTensor *arrModelInput = nullptr;
TfLiteTensor *arrModelOutput = nullptr;

const tflite::Model *segModel = nullptr;
tflite::MicroInterpreter *segInterpreter = nullptr;
TfLiteTensor *segModelInput = nullptr;
TfLiteTensor *segModelOutput = nullptr;

const tflite::Model *beatModel = nullptr;
tflite::MicroInterpreter *beatInterpreter = nullptr;
TfLiteTensor *beatModelInput = nullptr;
TfLiteTensor *beatModelOutput = nullptr;

int
init_models() {
    /**
     * @brief Initialize TFLM models
     *
     */
    size_t bytesUsed;
    TfLiteStatus allocateStatus;
    static tflite::AllOpsResolver opResolver;
    static tflite::MicroErrorReporter microErrorReporter;
    errorReporter = &microErrorReporter;

    tflite::InitializeTarget();

    // Load Arrhythmia model
    arrModel = tflite::GetModel(g_arrhythmia_model);
    if (arrModel->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(errorReporter, "Schema mismatch: given=%d != expected=%d.", arrModel->version(), TFLITE_SCHEMA_VERSION);
        return 1;
    }

    static tflite::MicroInterpreter arr_interpreter(arrModel, opResolver, arrTensorArena, arrTensorArenaSize, errorReporter);
    arrInterpreter = &arr_interpreter;

    allocateStatus = arrInterpreter->AllocateTensors();
    if (allocateStatus != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(errorReporter, "AllocateTensors() failed");
        return 1;
    }
    bytesUsed = arrInterpreter->arena_used_bytes();
    if (bytesUsed > arrTensorArenaSize) {
        TF_LITE_REPORT_ERROR(errorReporter, "Arena mismatch: given=%d < expected=%d bytes.", arrTensorArenaSize, bytesUsed);
        return 1;
    }
    arrModelInput = arrInterpreter->input(0);
    arrModelOutput = arrInterpreter->output(0);

    // Load Segmentation model
    segModel = tflite::GetModel(g_segmentation_model);
    if (segModel->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(errorReporter, "Schema mismatch: given=%d != expected=%d.", segModel->version(), TFLITE_SCHEMA_VERSION);
        return 1;
    }

    static tflite::MicroInterpreter seg_interpreter(segModel, opResolver, segTensorArena, segTensorArenaSize, errorReporter);
    segInterpreter = &seg_interpreter;

    allocateStatus = segInterpreter->AllocateTensors();
    if (allocateStatus != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(errorReporter, "AllocateTensors() failed");
        return 1;
    }
    bytesUsed = arrInterpreter->arena_used_bytes();
    if (bytesUsed > segTensorArenaSize) {
        TF_LITE_REPORT_ERROR(errorReporter, "Arena mismatch: given=%d < expected=%d bytes.", segTensorArenaSize, bytesUsed);
        return 1;
    }
    segModelInput = segInterpreter->input(0);
    segModelOutput = segInterpreter->output(0);

    // Load Beat interpreter
    beatModel = tflite::GetModel(g_beat_model);
    if (beatModel->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(errorReporter, "Schema mismatch: given=%d != expected=%d.", beatModel->version(), TFLITE_SCHEMA_VERSION);
        return 1;
    }

    static tflite::MicroInterpreter beat_interpreter(beatModel, opResolver, beatTensorArena, beatTensorArenaSize, errorReporter);
    beatInterpreter = &beat_interpreter;

    allocateStatus = beatInterpreter->AllocateTensors();
    if (allocateStatus != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(errorReporter, "AllocateTensors() failed");
        return 1;
    }

    bytesUsed = beatInterpreter->arena_used_bytes();
    if (bytesUsed > beatTensorArenaSize) {
        TF_LITE_REPORT_ERROR(errorReporter, "Arena mismatch: given=%d < expected=%d bytes.", beatTensorArenaSize, bytesUsed);
        return 1;
    }
    beatModelInput = beatInterpreter->input(0);
    beatModelOutput = beatInterpreter->output(0);

    return 0;
}

int
arrhythmia_inference(float32_t *x, float32_t threshold) {
    /**
     * @brief Run arrhythmia inference
     * @param x Model inputs
     * @param y Model outputs
     * @return Output label index
     */
    uint32_t yIdx = 0;
    float32_t yVal = 0;
    float32_t yMax = 0;

    // Quantize input
    for (int i = 0; i < arrModelInput->dims->data[1]; i++) {
        arrModelInput->data.int8[i] = x[i] / arrModelInput->params.scale + arrModelInput->params.zero_point;
    }

    // Invoke model
    TfLiteStatus invokeStatus = arrInterpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
        return -1;
    }

    // Dequantize output
    for (int i = 0; i < arrModelOutput->dims->data[1]; i++) {
        yVal = ((float32_t)arrModelOutput->data.int8[i] - arrModelOutput->params.zero_point) * arrModelOutput->params.scale;
        if ((i == 0) || (yVal > yMax)) {
            yMax = yVal;
            yIdx = i;
        }
    }
    return yIdx;
}

int
segmentation_inference(float32_t *data, int32_t *segMask, uint32_t padLen) {
    /**
     * @brief Run segmentation inference
     * @param data Model input
     * @param segMask Output segmentation mask
     * @return Success
     */
    uint32_t yIdx = 0;
    uint32_t yMaxIdx = 0;
    int32_t yVal = 0;
    int32_t yMax = 0;

    // Quantize input
    for (int i = 0; i < segModelInput->dims->data[1]; i++) {
        segModelInput->data.int8[i] = data[i] / segModelInput->params.scale + segModelInput->params.zero_point;
    }
    // Invoke model
    TfLiteStatus invokeStatus = segInterpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
        return -1;
    }
    // Dequantize output
    for (int i = padLen; i < segModelOutput->dims->data[1] - padLen; i++) {
        for (size_t j = 0; j < segModelOutput->dims->data[2]; j++) {
            yVal = ((int32_t)segModelOutput->data.int8[yIdx] - segModelOutput->params.zero_point) * segModelOutput->params.scale;
            yIdx += 1;
            if ((j == 0) || (yVal > yMax)) {
                yMax = yVal;
                yMaxIdx = j;
            }
        }
        segMask[i] = yMaxIdx;
    }
    return 0;
}

int
beat_inference(float32_t *pBeat, float32_t *beat, float32_t *nBeat) {
    /**
     * @brief Run beat inference
     * @param x Model inputs
     * @param y Model outputs
     * @return Output label index
     */
    uint32_t xIdx = 0;
    uint32_t yIdx = 0;
    int32_t yVal = 0;
    int32_t yMax = 0;

    // Quantize input
    for (int i = 0; i < beatModelInput->dims->data[2]; i++) {
        beatModelInput->data.int8[xIdx++] = pBeat[i] / beatModelInput->params.scale + beatModelInput->params.zero_point;
        beatModelInput->data.int8[xIdx++] = beat[i] / beatModelInput->params.scale + beatModelInput->params.zero_point;
        beatModelInput->data.int8[xIdx++] = nBeat[i] / beatModelInput->params.scale + beatModelInput->params.zero_point;
    }
    // Invoke model
    TfLiteStatus invokeStatus = beatInterpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
        return -1;
    }
    // Dequantize output
    for (int i = 0; i < beatModelOutput->dims->data[1]; i++) {
        yVal = ((int32_t)beatModelOutput->data.int8[i] - beatModelOutput->params.zero_point) * beatModelOutput->params.scale;
        if ((i == 0) || (yVal > yMax)) {
            yMax = yVal;
            yIdx = i;
        }
    }
    return yIdx;
}
