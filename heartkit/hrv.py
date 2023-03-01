from .defines import HeartExportParams, HeartTestParams, HeartTrainParams


def train_model(params: HeartTrainParams):
    """Train HRV model.

    Args:
        params (HeartTrainParams): Training parameters
    """
    # Load segmentation datasets
    # Load segmentation model
    # Compute HRV metrics across true and predicted segmentation masks
    # Routine
    #   Identify R peak from QRS segments and compute mean and std
    #   Remove outlier R peak regions based on confidence and std
    #   Compute heart rate, rhythm label, RR interval, RR variation


def evaluate_model(params: HeartTestParams):
    """Test HRV model.

    Args:
        params (HeartTestParams): Testing/evaluation parameters
    """


def export_model(params: HeartExportParams):
    """Export segmentation model.

    Args:
        params (HeartDemoParams): Deployment parameters
    """
