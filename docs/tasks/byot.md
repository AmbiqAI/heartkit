# Bring-Your-Own-Task (BYOT)

The Bring-Your-Own-Task (BYOT) feature allows users to create custom tasks for training, evaluating, and deploying heart-related AI models. This feature is useful for creating custom workflows for a given application with minimal coding.


## <span class="sk-h2-span">How it Works</span>

1. **Create a Task**: Define a new task by creating a new Python file. The file should contain a class that inherits from the `HKTask` base class and implements the required methods.

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

2. **Register the Task**: Register the new task with the `TaskFactory` by calling the `register` method. This method takes the task name and the task class as arguments.

    ```python
    ...

    hk.TaskFactory.register("custom", CustomTask)
    ```

3. **Use the Task**: The new task can now be used with the `TaskFactory` to perform various operations such as training, evaluating, and deploying models.

    ```python
    ...

    params = hk.HKTrainParams(...)
    task = hk.TaskFactory.get("custom")
    task.train(params)

    ```
