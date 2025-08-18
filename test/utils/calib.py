import torch

def get_calib(kspace: torch.Tensor, center_fraction: float = 0.08) -> torch.Tensor:
    """
    Extracts calibration region from k-space by keeping full sx
    and only a fraction of the center along sy.

    Parameters
    ----------
    kspace : torch.Tensor
        Input k-space of shape (coils, sx, sy).
    center_fraction : float
        Fraction of sy dimension to keep in the center (0 < f <= 1).

    Returns
    -------
    calib : torch.Tensor
        Calibration region of shape (coils, sx, sy_calib).
    """
    if kspace.ndim != 3:
        raise ValueError("Expected kspace shape (coils, sx, sy)")

    _, _, sy = kspace.shape

    # Compute size along phase-encode (sy)
    sy_calib = int(round(sy * center_fraction))
    sy_calib = max(sy_calib, 1)

    # Center index
    cy = sy // 2
    y_start, y_end = cy - sy_calib // 2, cy + (sy_calib + 1) // 2

    calib = kspace[:, :, y_start:y_end]

    return calib
