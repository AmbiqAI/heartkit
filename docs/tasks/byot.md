# Bring-Your-Own-Task (BYOT)

The Bring-Your-Own-Task (BYOT) feature allows users to create custom tasks for training, evaluating, and deploying heart-related AI models. This feature is useful for creating custom workflows for a given application with minimal coding.

For an in-depth guide check out the [BYOT Notebook Guide](../guides/byot.ipynb).

## <span class="sk-h2-span">How it Works</span>

1. **Create a Task**: Define a new task that inherits from the [HKTask](../api/tasks/task.md) base class and implements the required methods: `train`, `evaluate`, `export`, and `demo`.

    ```python
    import heartkit as hk

    class CustomTask(hk.HKTask):

        @staticmethod
        def train(params: hk.HKTrainParams) -> None:
            pass

        @staticmethod
        def evaluate(params: hk.HKTestParams) -> None:
            pass

        @staticmethod
        def export(params: hk.HKExportParams) -> None:
            pass

        @staticmethod
        def demo(params: hk.HKDemoParams) -> None:
            pass

    ```

2. **Register the Task**: Register the new task with the [TaskFactory](../api/tasks/factory.md) by calling the `register` method. This method takes the task name and the task class as arguments.

    ```python
    # Register the custom task
    hk.TaskFactory.register("custom", CustomTask)
    ```

3. **Use the Task**: The new task can now be used with the `TaskFactory` to perform various operations such as training, evaluating, and deploying models.

    ```python
    # Define task parameters
    params = hk.HKTrainParams(...)

    # Get the custom task
    task = hk.TaskFactory.get("custom")

    # Train the model
    task.train(params)

    ```
