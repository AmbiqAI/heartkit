/**
 * @file model.cc
 * @author Adam Page (adam.page@ambiq.com)
 * @brief Performs inference using TFLM
 * @version 1.0
 * @date 2023-03-27
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "model.h"
#include "arrhythmia_model_buffer.h"
#include "beat_model_buffer.h"
#include "constants.h"
#include "segmentation_model_buffer.h"

#include "ns_ambiqsuite_harness.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

//*****************************************************************************
//*** Tensorflow Globals
static tflite::ErrorReporter *errorReporter = nullptr;
constexpr int arrTensorArenaSize = 1024 * ARR_MODEL_SIZE_KB;
constexpr int segTensorArenaSize = 1024 * SEG_MODEL_SIZE_KB;
constexpr int beatTensorArenaSize = 1024 * BEAT_MODEL_SIZE_KB;
alignas(16) static uint8_t arrTensorArena[arrTensorArenaSize];
alignas(16) static uint8_t segTensorArena[segTensorArenaSize];
alignas(16) static uint8_t beatTensorArena[beatTensorArenaSize];

#ifdef ARRHTYHMIA_ENABLE
tf_model_config_t arrModel = {
    .arenaSize = arrTensorArenaSize,
    .arena = arrTensorArena,
    .buffer = g_arrhythmia_model,
    .model = nullptr,
    .input = nullptr,
    .output = nullptr,
    .interpreter = nullptr,
};
#endif

#ifdef SEGMENTATION_ENABLE
tf_model_config_t segModel = {
    .arenaSize = segTensorArenaSize,
    .arena = segTensorArena,
    .buffer = g_segmentation_model,
    .model = nullptr,
    .input = nullptr,
    .output = nullptr,
    .interpreter = nullptr,
};
#endif

#ifdef BEAT_ENABLE
tf_model_config_t beatModel = {
    .arenaSize = beatTensorArenaSize,
    .arena = beatTensorArena,
    .buffer = g_beat_model,
    .model = nullptr,
    .input = nullptr,
    .output = nullptr,
    .interpreter = nullptr,
};
#endif

uint32_t
init_models() {
    /**
     * @brief Initialize TFLM models
     * @return 0 if success else error code
     */
    size_t bytesUsed;
    TfLiteStatus allocateStatus;
    static tflite::AllOpsResolver opResolver;
    // ^ Use microOpResolver to reduce overhead
    static tflite::MicroErrorReporter microErrorReporter;
    errorReporter = &microErrorReporter;

    tflite::InitializeTarget();

    // Load Arrhythmia model
#if ARRHTYHMIA_ENABLE
    arrModel.model = tflite::GetModel(arrModel.buffer);
    if (arrModel.model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(errorReporter, "Schema mismatch: given=%d != expected=%d.", arrModel.model->version(), TFLITE_SCHEMA_VERSION);
        return 1;
    }
    static tflite::MicroInterpreter arr_interpreter(arrModel.model, opResolver, arrModel.arena, arrModel.arenaSize, errorReporter);
    arrModel.interpreter = &arr_interpreter;

    allocateStatus = arrModel.interpreter->AllocateTensors();
    if (allocateStatus != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(errorReporter, "AllocateTensors() failed");
        return 1;
    }
    bytesUsed = arrModel.interpreter->arena_used_bytes();
    if (bytesUsed > arrModel.arenaSize) {
        TF_LITE_REPORT_ERROR(errorReporter, "Arena mismatch: given=%d < expected=%d bytes.", arrModel.arenaSize, bytesUsed);
        return 1;
    }
    ns_printf("Arrhythmia needs %d bytes\n", bytesUsed);
    arrModel.input = arrModel.interpreter->input(0);
    arrModel.output = arrModel.interpreter->output(0);
#endif

    // Load Segmentation model
#if SEGMENTATION_ENABLE
    segModel.model = tflite::GetModel(segModel.buffer);
    if (segModel.model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(errorReporter, "Schema mismatch: given=%d != expected=%d.", segModel.model->version(), TFLITE_SCHEMA_VERSION);
        return 1;
    }

    static tflite::MicroInterpreter seg_interpreter(segModel.model, opResolver, segModel.arena, segModel.arenaSize, errorReporter);
    segModel.interpreter = &seg_interpreter;

    allocateStatus = segModel.interpreter->AllocateTensors();
    if (allocateStatus != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(errorReporter, "AllocateTensors() failed");
        return 1;
    }
    bytesUsed = segModel.interpreter->arena_used_bytes();
    if (bytesUsed > segModel.arenaSize) {
        TF_LITE_REPORT_ERROR(errorReporter, "Arena mismatch: given=%d < expected=%d bytes.", segModel.arenaSize, bytesUsed);
        return 1;
    }
    segModel.input = segModel.interpreter->input(0);
    segModel.output = segModel.interpreter->output(0);
    ns_printf("Segmentation model needs %d bytes\n", bytesUsed);
#endif

    // Load Beat interpreter
#if BEAT_ENABLE
    beatModel.model = tflite::GetModel(g_beat_model);
    if (beatModel.model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(errorReporter, "Schema mismatch: given=%d != expected=%d.", beatModel.model->version(), TFLITE_SCHEMA_VERSION);
        return 1;
    }

    static tflite::MicroInterpreter beat_interpreter(beatModel.model, opResolver, beatModel.arena, beatModel.arenaSize, errorReporter);
    beatModel.interpreter = &beat_interpreter;

    allocateStatus = beatModel.interpreter->AllocateTensors();
    if (allocateStatus != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(errorReporter, "AllocateTensors() failed");
        return 1;
    }

    bytesUsed = beatModel.interpreter->arena_used_bytes();
    if (bytesUsed > beatModel.arenaSize) {
        TF_LITE_REPORT_ERROR(errorReporter, "Arena mismatch: given=%d < expected=%d bytes.", beatModel.arenaSize, bytesUsed);
        return 1;
    }
    beatModel.input = beatModel.interpreter->input(0);
    beatModel.output = beatModel.interpreter->output(0);
    ns_printf("Beat model needs %d bytes\n", bytesUsed);
#endif
    return 0;
}

uint32_t
arrhythmia_inference(float32_t *x, float32_t *yVal, uint32_t *yIdx) {
    /**
     * @brief Run arrhythmia inference
     * @param x Model inputs
     * @param yVal Y output
     * @param yIdx Y class
     * @return 0 if success else error code
     */
    *yIdx = 0;
    *yVal = 0;
    float32_t yCur = 0;
#if ARRHTYHMIA_ENABLE
    // Quantize input
    for (int i = 0; i < arrModel.input->dims->data[2]; i++) {
        arrModel.input->data.int8[i] = x[i] / arrModel.input->params.scale + arrModel.input->params.zero_point;
    }

    // Invoke model
    TfLiteStatus invokeStatus = arrModel.interpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
        return invokeStatus;
    }

    // Dequantize output
    for (int i = 0; i < arrModel.output->dims->data[1]; i++) {
        yCur = ((float32_t)arrModel.output->data.int8[i] - arrModel.output->params.zero_point) * arrModel.output->params.scale;
        if ((i == 0) || (yCur > *yVal)) {
            *yVal = yCur;
            *yIdx = i;
        }
    }
#endif
    return 0;
}

uint32_t
segmentation_inference(float32_t *data, uint8_t *segMask, uint32_t padLen, float32_t threshold) {
    /**
     * @brief Run segmentation inference
     * @param data Model input
     * @param segMask Output segmentation mask
     * @param padLen Pad length of input to skip segment results
     * @return 0 if success else error code
     */
    uint32_t yIdx = 0;
    uint8_t yMaxIdx = 0;
    float32_t yVal = 0;
    float32_t yMax = 0;
#if (SEGMENTATION_ENABLE == 0)
    return 0;
#endif
    // Quantize input
    for (int i = 0; i < segModel.input->dims->data[2]; i++) {
        segModel.input->data.int8[i] = data[i] / segModel.input->params.scale + segModel.input->params.zero_point;
    }
    // Invoke model
    TfLiteStatus invokeStatus = segModel.interpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
        return invokeStatus;
    }
    // Dequantize output
    for (int i = padLen; i < segModel.output->dims->data[1] - (int)padLen; i++) {
        for (int j = 0; j < segModel.output->dims->data[2]; j++) {
            yIdx = i * segModel.output->dims->data[2] + j;
            yVal = ((float32_t)segModel.output->data.int8[yIdx] - segModel.output->params.zero_point) * segModel.output->params.scale;
            if ((j == 0) || (yVal > yMax)) {
                yMax = yVal;
                yMaxIdx = j;
            }
        }
        segMask[i] = yMaxIdx;
        // segMask[i] = yMax >= threshold ? yMaxIdx : 0;
    }
    return 0;
}

uint32_t
beat_inference(float32_t *pBeat, float32_t *beat, float32_t *nBeat, float32_t *yVal, uint32_t *yIdx) {
    /**
     * @brief Run beat inference
     * @param pBeat Previous beat input
     * @param beat Target beat input
     * @param nBeat Next beat input
     * @param yVal Y output
     * @param yIdx Y class
     * @return 0 if success else TfLiteStatus
     */
    uint32_t xIdx = 0;
    *yIdx = 0;
    *yVal = 0;
    float32_t yCur = 0;
#if BEAT_ENABLE
    // Quantize input
    for (int i = 0; i < beatModel.input->dims->data[2]; i++) {
        beatModel.input->data.int8[xIdx++] = pBeat[i] / beatModel.input->params.scale + beatModel.input->params.zero_point;
        beatModel.input->data.int8[xIdx++] = beat[i] / beatModel.input->params.scale + beatModel.input->params.zero_point;
        beatModel.input->data.int8[xIdx++] = nBeat[i] / beatModel.input->params.scale + beatModel.input->params.zero_point;
    }
    // Invoke model
    TfLiteStatus invokeStatus = beatModel.interpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
        return invokeStatus;
    }
    // Dequantize output
    for (int i = 0; i < beatModel.output->dims->data[1]; i++) {
        yCur = ((float32_t)beatModel.output->data.int8[i] - beatModel.output->params.zero_point) * beatModel.output->params.scale;
        if ((i == 0) || (yCur > *yVal)) {
            *yVal = yCur;
            *yIdx = i;
        }
    }
#endif
    return 0;
}
