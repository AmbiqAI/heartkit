# heartKIT Task Modes

## Introduction

Rather than offering a handful of static models, heartKIT provides a complete framework designed to cover the entire design process of creating customized ML models well-suited for low-power, wearable applications. Each mode serves a specific purpose and is engineered to offer you the flexibility and efficiency required for different tasks and use-cases.

Each `Task` implementes routines for each of the modes: [download](#download), [train](#train), [evaluate](#evaluate), [export](#export), and [demo](#demo). These modes are designed to streamline the process of training, evaluating, exporting, and running task-level demonstrations on the trained models.

---

## Available Modes

- **[Download](./download.md)**: Download specified datasets
- **[Train](./train.md)**: Train a model for specified task and datasets
- **[Evaluate](./evaluate.md)**: Evaluate a model for specified task and datasets
- **[Export](./export.md)**: Export a trained model to TensorFlow Lite and TFLM
- **[Demo](./demo.md)**: Run task-level demo on PC or remotely on Ambiq EVB

---

## [Download](./download.md)

[Download mode](./download.md) is used to download the specified datasets for the task. The routine can be customized via the configuration file or by setting the parameters directly in the code. The download process involves fetching the dataset(s) from the specified source and storing them in the specified directory.

## [Train](./train.md)

[Train mode](./train.md) is used to train a model for the specified task and dataset. In this mode, the model is trained for a given task using the specified dataset(s), model architecture, and hyperparameters. The training process involves optimizing the model's parameters to maximize its performance on the training data.

## [Evaluate](./evaluate.md)

[Evaluate mode](./evaluate.md) is used to test the performance of the model on the reserved test set for the specified task. The routine can be customized via the configuration file or by setting the parameters directly in the code. The evaluation process involves testing the model's performance on the test data to measure its accuracy, precision, recall, and F1 score.

## [Export](./export.md)

[Export mode](./export.md) is used to convert the trained model into a format that can be used for deployment onto Ambiq's family of SoCs. Currently, the command will convert the TensorFlow model into both TensorFlow Lite (TFL) and TensorFlow Lite for micro-controller (TFLM) variants. The command will also verify the models' outputs match.

## [Demo](./demo.md)

[Demo mode](./demo.md) is used to run a task-level demonstration on the trained model using the specified backend inference engine (e.g. PC or EVB). This is useful to showcase the model's performance in real-time and to verify its accuracy in a real-world scenario.

---

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
