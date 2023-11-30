from ..defines import HeartSegment


def get_classes(nclasses: int = 4) -> list[str]:
    """Get classes

    Args:
        nclasses (int): Number of classes

    Returns:
        list[str]: List of class names
    """
    if 2 <= nclasses <= 5:
        return list(range(nclasses))
    raise ValueError(f"Invalid number of classes: {nclasses}")


def get_class_mapping(nclasses: int = 4) -> dict[int, int]:
    """Get class mapping

    Args:
        nclasses (int): Number of classes

    Returns:
        dict[int, int]: Class mapping
    """
    match nclasses:
        case 2:
            return {
                HeartSegment.normal: 0,
                HeartSegment.pwave: 0,
                HeartSegment.qrs: 1,
                HeartSegment.twave: 0,
                HeartSegment.uwave: 0,
            }
        case 3:
            return {
                HeartSegment.normal: 0,
                HeartSegment.pwave: 2,
                HeartSegment.qrs: 1,
                HeartSegment.twave: 2,
                HeartSegment.uwave: 0,
            }
        case 4:
            return {
                HeartSegment.normal: 0,
                HeartSegment.pwave: 1,
                HeartSegment.qrs: 2,
                HeartSegment.twave: 3,
                HeartSegment.uwave: 0,
            }
        case 5:
            return {
                HeartSegment.normal: 0,
                HeartSegment.pwave: 1,
                HeartSegment.qrs: 2,
                HeartSegment.twave: 3,
                HeartSegment.uwave: 4,
            }
        case _:
            raise ValueError(f"Invalid number of classes: {nclasses}")
    # END MATCH


def get_class_names(nclasses: int = 4) -> list[str]:
    """Get class names

    Args:
        nclasses (int): Number of classes

    Returns:
        list[str]: List of class names
    """
    match nclasses:
        case 2:
            return ["NONE", "QRS"]
        case 3:
            return ["NONE", "QRS", "P/T-WAVE"]
        case 4:
            return ["NONE", "P-WAVE", "QRS", "T-WAVE"]
        case 5:
            return ["NONE", "P-WAVE", "QRS", "T-WAVE", "U-WAVE"]
        case _:
            raise ValueError(f"Invalid number of classes: {nclasses}")
