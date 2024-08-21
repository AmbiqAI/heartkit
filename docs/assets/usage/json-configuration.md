```javascript
{
    "name": "arr-2-eff-sm",
    "project": "hk-rhythm-2",
    "job_dir": "./results/arr-2-eff-sm",
    "verbose": 2,
    "datasets": [
        {
            "name": "ptbxl",
            "params": {
                "path": "./datasets/ptbxl"
            }
        }
    ],
    "num_classes": 2,
    "class_map": {
        "0": 0,
        "7": 1,
        "8": 1
    },
    "class_names": [
        "NORMAL",
        "AFIB/AFL"
    ],
    "class_weights": "balanced",
    "sampling_rate": 100,
    "frame_size": 512,
    "samples_per_patient": [
        10,
        10
    ],
    "val_samples_per_patient": [
        5,
        5
    ],
    "test_samples_per_patient": [
        5,
        5
    ],
    "val_patients": 0.2,
    "val_size": 20000,
    "test_size": 20000,
    "batch_size": 256,
    "buffer_size": 20000,
    "epochs": 100,
    "steps_per_epoch": 50,
    "val_metric": "loss",
    "lr_rate": 0.001,
    "lr_cycles": 1,
    "threshold": 0.75,
    "val_metric_threshold": 0.98,
    "tflm_var_name": "g_rhythm_model",
    "tflm_file": "rhythm_model_buffer.h",
    "backend": "pc",
    "demo_size": 896,
    "display_report": true,
    "quantization": {
        "qat": false,
        "format": "INT8",
        "io_type": "int8",
        "conversion": "CONCRETE",
        "debug": false
    },
    "preprocesses": [
        {
            "name": "layer_norm",
            "params": {
                "epsilon": 0.01,
                "name": "znorm"
            }
        }
    ],
    "augmentations": [],
    "model_file": "model.keras",
    "use_logits": false,
    "architecture": {
        "name": "efficientnetv2",
        "params": {
            "input_filters": 16,
            "input_kernel_size": [
                1,
                9
            ],
            "input_strides": [
                1,
                2
            ],
            "blocks": [
                {
                    "filters": 24,
                    "depth": 2,
                    "kernel_size": [
                        1,
                        9
                    ],
                    "strides": [
                        1,
                        2
                    ],
                    "ex_ratio": 1,
                    "se_ratio": 2
                },
                {
                    "filters": 32,
                    "depth": 2,
                    "kernel_size": [
                        1,
                        9
                    ],
                    "strides": [
                        1,
                        2
                    ],
                    "ex_ratio": 1,
                    "se_ratio": 2
                },
                {
                    "filters": 40,
                    "depth": 2,
                    "kernel_size": [
                        1,
                        9
                    ],
                    "strides": [
                        1,
                        2
                    ],
                    "ex_ratio": 1,
                    "se_ratio": 2
                },
                {
                    "filters": 48,
                    "depth": 1,
                    "kernel_size": [
                        1,
                        9
                    ],
                    "strides": [
                        1,
                        2
                    ],
                    "ex_ratio": 1,
                    "se_ratio": 2
                }
            ],
            "output_filters": 0,
            "include_top": true,
            "use_logits": true
        }
    }
}
```
