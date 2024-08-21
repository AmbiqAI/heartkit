```python

hk.HKTaskParams(
    name="arr-2-eff-sm",
    project="hk-rhythm-2",
    job_dir="./results/arr-2-eff-sm",
    verbose=2,
    datasets=[hk.NamedParams(
        name="ptbxl",
        params=dict(
            path="./datasets/ptbxl"
        )
    )],
    num_classes=2,
    class_map={
        "0": 0,
        "7": 1,
        "8": 1
    },
    class_names=[
        "NORMAL", "AFIB/AFL"
    ],
    class_weights="balanced",
    sampling_rate=100,
    frame_size=512,
    samples_per_patient=[10, 10],
    val_samples_per_patient=[5, 5],
    test_samples_per_patient=[5, 5],
    val_patients=0.20,
    val_size=20000,
    test_size=20000,
    batch_size=256,
    buffer_size=20000,
    epochs=100,
    steps_per_epoch=50,
    val_metric="loss",
    lr_rate=1e-3,
    lr_cycles=1,
    threshold=0.75,
    val_metric_threshold=0.98,
    tflm_var_name="g_rhythm_model",
    tflm_file="rhythm_model_buffer.h",
    backend="pc",
    demo_size=896,
    display_report=True,
    quantization=hk.QuantizationParams(
        qat=False,
        format="INT8",
        io_type="int8",
        conversion="CONCRETE",
        debug=False
    ),
    preprocesses=[
        hk.NamedParams(
            name="layer_norm",
            params=dict(
                epsilon=0.01,
                name="znorm"
            )
        )
    ],
    augmentations=[
    ],
    model_file="model.keras",
    use_logits=False,
    architecture=hk.NamedParams(
        name="efficientnetv2",
        params=dict(
            input_filters=16,
            input_kernel_size=[1, 9],
            input_strides=[1, 2],
            blocks=[
                {"filters": 24, "depth": 2, "kernel_size": [1, 9], "strides": [1, 2], "ex_ratio": 1,  "se_ratio": 2},
                {"filters": 32, "depth": 2, "kernel_size": [1, 9], "strides": [1, 2], "ex_ratio": 1,  "se_ratio": 2},
                {"filters": 40, "depth": 2, "kernel_size": [1, 9], "strides": [1, 2], "ex_ratio": 1,  "se_ratio": 2},
                {"filters": 48, "depth": 1, "kernel_size": [1, 9], "strides": [1, 2], "ex_ratio": 1,  "se_ratio": 2}
            ],
            output_filters=0,
            include_top=True,
            use_logits=True
        )
    }
)
```
