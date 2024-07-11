import numpy.typing as npt


SpecType = list[npt.ArrayLike] | tuple[npt.ArrayLike] | dict[str, npt.ArrayLike] | npt.ArrayLike


def matches_spec(o: SpecType, spec: SpecType, ignore_batch_dim: bool = False) -> bool:
    """Test whether data object matches the desired spec.

    Args:
        o (SpecType): Data object.
        spec (SpecType): Metadata for describing the the data object.
        ignore_batch_dim: Ignore first dimension when checking shapes.

    Returns:
        bool: If matches
    """
    if isinstance(spec, (list, tuple)):
        if not isinstance(o, (list, tuple)):
            raise ValueError(f"data object is not a list or tuple which is required by the spec: {spec}")
        if len(spec) != len(o):
            raise ValueError(f"data object has a different number of elements than the spec: {spec}")
        for i, ispec in enumerate(spec):
            if not matches_spec(o[i], ispec, ignore_batch_dim=ignore_batch_dim):
                return False
        return True

    if isinstance(spec, dict):
        if not isinstance(o, dict):
            raise ValueError(f"data object is not a dict which is required by the spec: {spec}")
        if spec.keys() != o.keys():
            raise ValueError(f"data object has different keys than those specified in the spec: {spec}")
        for k in spec:
            if not matches_spec(o[k], spec[k], ignore_batch_dim=ignore_batch_dim):
                return False
            return True

    spec_shape = spec.shape[1:] if ignore_batch_dim else spec.shape
    o_shape = o.shape[1:] if ignore_batch_dim else o.shape
    return spec_shape == o_shape and spec.dtype == o.dtype
