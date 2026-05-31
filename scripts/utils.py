import re
import numpy as np
from pathlib import Path

# ============================================================================
# VALID VERTEX MASK UTILITIES
# ============================================================================
# Default path to the valid vertex mask
DEFAULT_MASK_PATH = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/valid_vertex_mask.npy")

_cached_mask = None  # Cache the mask to avoid repeated disk reads


def load_valid_vertex_mask(mask_path: str | Path = None) -> np.ndarray:
    """
    Load the valid vertex mask (boolean array where True = valid vertex).
    
    The mask excludes medial wall / zero-variance vertices that have no BOLD signal.
    
    Args:
        mask_path: Path to the mask file. If None, uses the default path.
    
    Returns:
        Boolean array of shape (64984,) where True indicates a valid vertex.
    """
    global _cached_mask
    
    if mask_path is None:
        mask_path = DEFAULT_MASK_PATH
    else:
        mask_path = Path(mask_path)
    
    # Use cached mask if available
    if _cached_mask is not None:
        return _cached_mask
    
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Valid vertex mask not found at {mask_path}. "
            "Run generate_valid_vertex_mask.py first."
        )
    
    _cached_mask = np.load(mask_path)
    return _cached_mask


def apply_vertex_mask(data: np.ndarray, mask: np.ndarray = None, axis: int = -1) -> np.ndarray:
    """
    Apply the valid vertex mask to data, keeping only valid vertices.
    
    Args:
        data: Array containing vertex data. The vertex dimension is specified by 'axis'.
        mask: Boolean mask array. If None, loads the default mask.
        axis: Axis along which to apply the mask (default: -1, the last axis).
    
    Returns:
        Data with invalid vertices removed along the specified axis.
    """
    if mask is None:
        mask = load_valid_vertex_mask()
    
    # Move axis to the end for easier indexing
    data = np.moveaxis(data, axis, -1)
    
    # Apply mask
    data_masked = data[..., mask]
    
    # Move axis back
    data_masked = np.moveaxis(data_masked, -1, axis)
    
    return data_masked


def apply_mask_to_betas_dict(betas_dict: dict, mask: np.ndarray = None) -> dict:
    """
    Apply the valid vertex mask to a dictionary of beta arrays.
    
    The input dictionary typically has structure:
        {condition_key: array of shape (n_trials, n_vertices)}
    
    Args:
        betas_dict: Dictionary mapping condition keys to beta arrays.
        mask: Boolean mask array. If None, loads the default mask.
    
    Returns:
        New dictionary with masked beta arrays (invalid vertices removed).
    """
    if mask is None:
        mask = load_valid_vertex_mask()
    
    masked_dict = {}
    for key, betas in betas_dict.items():
        if isinstance(betas, np.ndarray):
            # Assume shape is (n_trials, n_vertices) or (n_vertices,)
            if betas.ndim == 2:
                masked_dict[key] = betas[:, mask]
            elif betas.ndim == 1:
                masked_dict[key] = betas[mask]
            else:
                # For higher-dimensional arrays, assume last axis is vertices
                masked_dict[key] = apply_vertex_mask(betas, mask, axis=-1)
        elif isinstance(betas, list):
            # Handle list of arrays
            masked_dict[key] = [b[:, mask] if b.ndim == 2 else b[mask] for b in betas]
        else:
            masked_dict[key] = betas
    
    return masked_dict


def get_n_valid_vertices() -> int:
    """Return the number of valid vertices."""
    mask = load_valid_vertex_mask()
    return int(np.sum(mask))


def get_valid_vertex_indices() -> np.ndarray:
    """Return the indices of valid vertices."""
    mask = load_valid_vertex_mask()
    return np.where(mask)[0]


# ============================================================================
# RUN LABEL UTILITIES
# ============================================================================

def standardize_run_label(run_str):
    """
    Converts 'run-1' to 'run-01', 'run-10' stays as 'run-10', etc.
    """
    match = re.match(r"run-(\d+)", run_str)
    if match:
        run_num = int(match.group(1))
        return f"run-{run_num:02d}"
    else:
        raise ValueError(f"Invalid run format: {run_str}")


def glm_confounds_construction(df_confounds):
    # note this is a hand selection of good confounds (IMO) for task GLMs
    timeseries_confounds = []
    # csf signals
    timeseries_confounds.append(df_confounds['csf'])
    timeseries_confounds.append(df_confounds['csf_power2'])
    timeseries_confounds.append(df_confounds['csf_derivative1']) # nan
    timeseries_confounds.append(df_confounds['csf_derivative1_power2']) # nan
    # wm signals
    timeseries_confounds.append(df_confounds['white_matter'])
    timeseries_confounds.append(df_confounds['white_matter_power2'])
    timeseries_confounds.append(df_confounds['white_matter_derivative1']) # nan
    timeseries_confounds.append(df_confounds['white_matter_derivative1_power2']) # nan
    # motion signals
    timeseries_confounds.append(df_confounds['trans_x'])
    timeseries_confounds.append(df_confounds['trans_x_power2'])
    timeseries_confounds.append(df_confounds['trans_x_derivative1'])
    timeseries_confounds.append(df_confounds['trans_x_derivative1_power2'])
    timeseries_confounds.append(df_confounds['trans_y'])
    timeseries_confounds.append(df_confounds['trans_y_power2'])
    timeseries_confounds.append(df_confounds['trans_y_derivative1'])
    timeseries_confounds.append(df_confounds['trans_y_derivative1_power2'])
    timeseries_confounds.append(df_confounds['trans_z'])
    timeseries_confounds.append(df_confounds['trans_z_power2'])
    timeseries_confounds.append(df_confounds['trans_z_derivative1'])
    timeseries_confounds.append(df_confounds['trans_z_derivative1_power2'])
    timeseries_confounds.append(df_confounds['rot_x'])
    timeseries_confounds.append(df_confounds['rot_x_power2'])
    timeseries_confounds.append(df_confounds['rot_x_derivative1'])
    timeseries_confounds.append(df_confounds['rot_x_derivative1_power2'])
    timeseries_confounds.append(df_confounds['rot_y'])
    timeseries_confounds.append(df_confounds['rot_y_power2'])
    timeseries_confounds.append(df_confounds['rot_y_derivative1'])
    timeseries_confounds.append(df_confounds['rot_y_derivative1_power2'])
    timeseries_confounds.append(df_confounds['rot_z'])
    timeseries_confounds.append(df_confounds['rot_z_power2'])
    timeseries_confounds.append(df_confounds['rot_z_derivative1'])
    timeseries_confounds.append(df_confounds['rot_z_derivative1_power2'])
    #
    timeseries_confounds = np.asarray(timeseries_confounds).T # change to time x confounds
    return timeseries_confounds