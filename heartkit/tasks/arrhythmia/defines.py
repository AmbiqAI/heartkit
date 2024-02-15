from ...defines import HeartRhythm


def get_classes(nclasses: int = 2) -> list[str]:
    """Get classes

    Args:
        nclasses (int): Number of classes

    Returns:
        list[str]: List of class names
    """
    if 2 <= nclasses <= 3:
        return list(range(nclasses))
    raise ValueError(f"Invalid number of classes: {nclasses}")


def get_class_mapping(nclasses: int = 2) -> dict[int, int]:
    """Get class mapping

    Args:
        nclasses (int): Number of classes

    Returns:
        dict[int, int]: Class mapping
    """
    match nclasses:
        case 2:
            return {
                HeartRhythm.normal: 0,
                HeartRhythm.afib: 1,
                HeartRhythm.aflut: 1,
            }
        case 3:
            return {
                HeartRhythm.normal: 0,
                HeartRhythm.afib: 1,
                HeartRhythm.aflut: 2,
            }
        case _:
            raise ValueError(f"Invalid number of classes: {nclasses}")


def get_class_names(nclasses: int = 2) -> list[str]:
    """Get class names

    Args:
        nclasses (int): Number of classes

    Returns:
        list[str]: List of class names
    """
    match nclasses:
        case 2:
            return ["NSR", "AFIB/AFL"]
        case 3:
            return ["NSR", "AFIB", "AFL"]
        case _:
            raise ValueError(f"Invalid number of classes: {nclasses}")


def get_feat_shape(frame_size: int) -> tuple[int, ...]:
    """Get dataset feature shape.

    Args:
        frame_size (int): Frame size

    Returns:
        tuple[int, ...]: Feature shape
    """
    return (frame_size, 1)  # Time x Channels


def get_class_shape(frame_size: int, nclasses: int) -> tuple[int, ...]:
    """Get dataset class shape.

    Args:
        frame_size (int): Frame size
        nclasses (int): Number of classes

    Returns:
        tuple[int, ...]: Class shape
    """
    return (nclasses,)  # One-hot encoded classes
