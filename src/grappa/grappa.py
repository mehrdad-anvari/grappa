"""
Written by Mehrdad Anvari-Fard, Mahdi Bazargani, Mohammad Javad Heidari
to accelerate the grappa implementation in the pygrappa repository owned
by Nicholas McKibben using pytorch GPU capabilities and some other tricks.

The code is still under development!!
"""
import torch
import time

def unravel_index(flat_index, shape):
    # Ensure flat_index is a tensor
    if not isinstance(flat_index, torch.Tensor):
        flat_index = torch.tensor(flat_index, dtype=torch.int64)
    
    # Ensure shape is a tensor
    if not isinstance(shape, torch.Tensor):
        shape = torch.tensor(shape, dtype=torch.int64)
    
    # Initialize the result with zeros
    result = []
    
    # Calculate unravel indices
    for dim in reversed(shape):
        result.append(flat_index % dim)
        flat_index = flat_index // dim
    
    # Reverse the result to match the correct order
    return tuple(result[::-1])

def view_as_windows(arr_in, window_shape, step=1):
    # Check if the input is a tensor, if not convert it
    if not isinstance(arr_in, torch.Tensor):
        arr_in = torch.tensor(arr_in)

    # Ensure window_shape and step are tuples
    if isinstance(window_shape, int):
        window_shape = (window_shape,) * arr_in.ndim
    if not (len(window_shape) == arr_in.ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, int):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * arr_in.ndim

    if len(step) != arr_in.ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    # Convert shapes and strides to numpy arrays for easier manipulation
    arr_shape = torch.tensor(arr_in.shape)
    window_shape = torch.tensor(window_shape, dtype=torch.int)
    step = torch.tensor(step, dtype=torch.int)

    # Ensure window_shape is not larger than arr_in in any dimension
    if torch.any(arr_shape < window_shape):
        raise ValueError("`window_shape` is too large")

    if torch.any(window_shape < 1):
        raise ValueError("`window_shape` is too small")

    # Compute the new shape and strides
    win_indices_shape = (arr_shape - window_shape) // step + 1
    new_shape = tuple(win_indices_shape.tolist()) + tuple(window_shape.tolist())
    new_strides = tuple((torch.tensor(arr_in.stride()) * step).tolist()) + tuple(arr_in.stride())

    # Use as_strided to create the view
    arr_out = torch.as_strided(arr_in, size=new_shape, stride=new_strides)
    return arr_out

def torch_setdiff(t1: torch.Tensor,
                  t2: torch.Tensor) -> torch.Tensor:
    """
    calculate the 2 way difference for the 2 given tensor
    """
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    return difference
 
def find_unique_patterns(kspace: torch.Tensor,
                         kernel_size: int,
                         coil_axis: int=-1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function can be employed to find unique pattern once (for the first slice)
    and then used for other slices to avoid wasting time.
    """
    if len(kspace.shape) != 3:
        raise ValueError("`kspace` should be 3D: (width, height, coil)")
    kspace = torch.moveaxis(kspace, coil_axis, -1)
    mask = torch.abs(kspace[...,0])>0
    print(mask.shape)
    if mask.sum().item() == 0:
        raise ValueError("no acquire sample detected!")
    if (mask != 0).sum().item() == mask.shape[0] * mask.shape[1]:
        raise ValueError("The kspace is fully sampled!")
    
    # Extract patches and reshape for uniqueness check
    P = view_as_windows(mask, (kernel_size, kernel_size))
    P = P.reshape((-1, kernel_size, kernel_size))

    # Find unique patterns
    P_unique, iidx = torch.unique(P, return_inverse=True, dim=0)

    return P_unique, iidx

def pattern_matching(T: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """
    Finds the inverse indices for T where the pattern P[i,:,:] matches. 
    If any pattern does not match, it is indicated by -1 as inverse index. 
    T should have the shape of (N,kx,ky) and P should have the shape of (M,kx,ky) 
    where M is smaller than the N.
    """
    N, kx, ky = T.shape
    M, _, _ = P.shape
    
    # Flatten the patterns and tensor slices
    T_flat = T.view(N, -1)
    P_flat = P.view(M, -1)
    
    # Initialize an array to store the inverse indices
    reverse_indices = torch.full((N,), -1, dtype=torch.int64).to(T.device)
    
    # Use broadcasting and comparison to find matches
    for i in range(M):
        matches = torch.all(T_flat == P_flat[i], dim=1)
        reverse_indices[matches] = i
    
    return reverse_indices

def grappa(kspace: torch.Tensor, 
           calib: torch.Tensor, 
           kernel_size: tuple=(5, 5), 
           coil_axis: int=-1,
           lamda: float=0.01,
           undersampling_pattern: str="2D",
           unique_patterns: torch.Tensor=None):
    '''GeneRalized Autocalibrating Partially Parallel Acquisitions.

    Parameters
    ----------
    kspace : array_like
        2D multi-coil k-space data to reconstruct from.  Make sure
        that the missing entries have exact zeros in them.
    calib : array_like
        Calibration data (fully sampled k-space).
    kernel_size : tuple, optional
        Size of the 2D GRAPPA kernel (kx, ky).
    coil_axis : int, optional
        Dimension holding coil data.  The other two dimensions should
        be image size: (sx, sy).
    lamda : float, optional
        Tikhonov regularization for the kernel calibration.
    undersampling_pattern: str "1D" or "2D"
        used to modify the way to find unique undersampling patterns.
        choosing "1D" can greatly increases the speed if the undersampling
        is only done along one dimension.
    unique_patterns: torch.Tensor
        A tensor with the shape of (M,kx,ky) used to avoid using torch.uniqe
        which take too much time to execute. if the undersampling patterns
        are known then they can be given to the function to speed up the process.

    Returns
    -------
    res : array_like
        k-space data where missing entries have been filled in.

    References
    ----------
    .. [1] Griswold, Mark A., et al. "Generalized autocalibrating
           partially parallel acquisitions (GRAPPA)." Magnetic
           Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           47.6 (2002): 1202-1210.

    '''
    # Put the coil dimension at the end
    device = kspace.device
    kspace = torch.moveaxis(kspace, coil_axis, -1)
    calib = torch.moveaxis(calib, coil_axis, -1)
    
    # Get shape of kernel
    kx, ky = kernel_size[:]
    kx2, ky2 = int(kx/2), int(ky/2)
    nc = calib.shape[-1]

    # Notice that all coils have same sampling pattern, so choose
    # the 0th one arbitrarily for the mask
    mask = torch.abs(kspace[..., 0]) > 0

    # Quit early if there are no holes
    t1_start = time.perf_counter()
    if (mask).all():
        return torch.moveaxis(kspace, -1, coil_axis)
    
    P = view_as_windows(mask, (kx, ky))
    Psh = P.shape[:]  # save shape for unflattening indices later
    P = P.reshape((-1, kx, ky))

    # Find the unique patches and associate them with indices
    
    if unique_patterns is None:
        if undersampling_pattern == "2D":
            P, iidx = torch.unique(P, return_inverse=True, dim=0)
        elif undersampling_pattern == "1D":
            P2 = view_as_windows(mask[0:kx], (kx, ky))
            P2 = P2.reshape((-1, kx, ky))
            P, iidx = torch.unique(P2, return_inverse=True, dim=0)
            num_repeat = (mask.shape[0] - kx + 1)
            iidx = iidx.repeat(num_repeat)
        else:
            raise ValueError("`undersampling_pattern` should be either '1D' or '2D'")
    else:
        iidx = pattern_matching(P,unique_patterns)
        P = unique_patterns
    # Filter out geometries that don't have a hole at the center.
    # These are all the kernel geometries we actually need to
    # compute weights for.
    validP = torch.nonzero(~P[:, kx2, ky2]).squeeze()

    # We also want to ignore empty patches
    invalidP = torch.nonzero(torch.all(P.flatten(start_dim = 1) == 0, dim=1)).squeeze(dim=1)
    validP = torch_setdiff(validP, invalidP)

    # Give P back its coil dimension
    P = P.unsqueeze(-1).repeat(1, 1, 1, nc)

    kspace_windows = view_as_windows(kspace, (kx,ky,nc)).reshape((-1, kx, ky, nc))
    A = view_as_windows(calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))
    t1_end = time.perf_counter()
    t_unique = t1_end - t1_start

    t2_total, t3_total = 0.0, 0.0
    for ii in validP:
        # Step 2: Kernel calibration (weights)
        t2_start = time.perf_counter()
        S = A[:, P[ii, ...]]
        T = A[:, kx2, ky2, :]
        ShS = S.conj().T @ S
        ShT = S.conj().T @ T
        lamda0 = (lamda*torch.linalg.norm(ShS)/ShS.shape[0])
        W = torch.linalg.solve(
            ShS + lamda0*torch.eye(ShS.shape[0]).to(device), ShT)
        t2_end = time.perf_counter()
        t2_total += (t2_end - t2_start)

        # Step 3: Estimation of missing samples
        t3_start = time.perf_counter()
        idx_1d = torch.nonzero(iidx == ii)
        idx = unravel_index(idx_1d, Psh[:2])
        x, y = idx[0]+kx2, idx[1]+ky2
        
        x = x.squeeze()
        y = y.squeeze()

        S = kspace_windows[idx_1d,P[ii,...]]
        
        kspace[x,y] = S @ W
        t3_end = time.perf_counter()
        t3_total += (t3_end - t3_start)
        
    return torch.moveaxis(kspace, -1, coil_axis), (t_unique, t2_total, t3_total)

if __name__ == '__main__':
    pass
