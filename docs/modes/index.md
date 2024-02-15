# HeartKit Modes


## <span class="sk-h2-span">Introduction</span>

Rather than offering a handful of static models, HeartKit provides a complete framework designed to cover the entire design process of creating customized ML models well-suited for low-power, wearable applications. Each mode serves a specific purpose and is engineered to offer you the flexibility and efficiency required for different tasks and use-cases.


## Available Modes

- **[Download](./download.md)**: Download specified datasets
- **[Train](./train.md)**: Train a model for specified task and datasets
- **[Evaluate](./evaluate.md)**: Evaluate a model for specified task and datasets
- **[Export](./export.md)**: Export a trained model to TensorFlow Lite and TFLM
- **[Demo](./demo.md)**: Run task-level demo on PC or remotely on Ambiq EVB

## <span class="sk-h2-span">[Train](./train.md)</span>

Train mode is used to train a model for the specified task and dataset. In this mode, the model is trained for a given task using the specified dataset(s), model architecture, and hyperparameters. The training process involves optimizing the model's parameters to maximize its performance on the training data.

## <span class="sk-h2-span">[Evaluate](./evaluate.md)</span>

Evaluate mode is used to test the performance of the model on the reserved test set for the specified task. For certain tasks, a confidence threshold can also be set such that a label is only assigned when the model's probability is greater than the threshold; otherwise, a label of inconclusive will be assigned. This is useful in noisy environments where the model may not be confident in its prediction.

## <span class="sk-h2-span">[Export](./export.md)</span>

Export mode is used to convert the trained model into a format that can be used for deployment onto Ambiq's family of SoCs. Currently, the command will convert the TensorFlow model into both TensorFlow Lite (TFL) and TensorFlow Lite for micro-controller (TFLM) variants. The command will also verify the models' outputs match.

## <span class="sk-h2-span">[Demo](./demo.md)</span>

Demo mode is used to run a task-level demonstration using either the PC or EVB as backend inference engine. This is useful to showcase the model's performance in real-time and to verify its accuracy in a real-world scenario.


!!! Example "At-a-Glance"

    === "Download"

        <br>
        Download specified datasets. <br>
        Refer to [Download Mode](./download.md) for more details.

    === "Train"

        <br>
        Train a model for specified task and dataset(s). <br>
        Refer to [Train Mode](./train.md) for more details.

    === "Evaluate"

        <br>
        Evaluate a model for specified task and dataset(s). <br>
        Refer to [Evaluate Mode](./evaluate.md) for more details.

    === "Export"

        <br>
        Export a trained model to TensorFlow Lite and TFLM. <br>
        Refer to [Export Mode](./export.md) for more details.

    === "Demo"

        <br>
        Run task-level demo on PC or EVB. <br>
        Refer to [Demo Mode](./demo.md) for more details.
