from ...defines import HeartBeat


def get_classes(nclasses: int = 3) -> list[str]:
    """Get classes

    Args:
        nclasses (int): Number of classes

    """
    if 2 <= nclasses <= 3:
        return list(range(nclasses))
    raise ValueError(f"Invalid number of classes: {nclasses}")


def get_class_mapping(nclasses: int = 3) -> dict[int, int]:
    """Get class mapping

    Args:
        nclasses (int): Number of classes

    Returns:
        dict[int, int]: Class mapping
    """
    match nclasses:
        case 2:
            return {
                HeartBeat.normal: 0,
                HeartBeat.pac: 1,
                HeartBeat.pvc: 1,
            }
        case 3:
            return {
                HeartBeat.normal: 0,
                HeartBeat.pac: 1,
                HeartBeat.pvc: 2,
            }
        case _:
            raise ValueError(f"Invalid number of classes: {nclasses}")
    # END MATCH


def get_class_names(nclasses: int = 3) -> list[str]:
    """Get class names

    Args:
        nclasses (int): Number of classes

    """
    match nclasses:
        case 2:
            return ["QRS", "PAC/PVC"]
        case 3:
            return ["QRS", "PAC", "PVC"]
        case _:
            raise ValueError(f"Invalid number of classes: {nclasses}")


def get_feat_shape(frame_size: int) -> tuple[int, ...]:
    """Get dataset feature shape.

    Args:
        frame_size (int): Frame size

    Returns:
        tuple[int, ...]: Feature shape
    """
    return (frame_size, 3)  # Time x Channels


def get_class_shape(frame_size: int, nclasses: int) -> tuple[int, ...]:
    """Get dataset class shape.

    Args:
        frame_size (int): Frame size
        nclasses (int): Number of classes

    Returns:
        tuple[int, ...]: Class shape
    """
    return (nclasses,)  # One-hot encoded classes
