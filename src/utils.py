"""
File with supporting methods and classes.

Author: Till Schnabel (contact till.schnabel@inf.ethz.ch)

MIT License

Copyright (c) 2026 ETH Zurich, Till Schnabel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from datetime import datetime
import re
import warnings
from pathlib import Path
from typing import Union, Tuple, List, TYPE_CHECKING, Callable, Iterable
import contextlib

if TYPE_CHECKING:
    from src.objects.mesh import Mesh
import numpy as np

try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    # Fallback dummy function that just returns the iterable, in case tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument.
    Copied from https://stackoverflow.com/a/58936697/12232644.
    """
    import joblib

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def parallel_method(method: Callable, iterator: Iterable, iterator_name: str = None, num_jobs: int = 0,
                    tqdm_message: str = "Parallel Processing", verbose: bool = True,
                    start_from: str = None, stop_after: str = None, unpack_iterator: bool = False,
                    total: int = None, backend: str = None, batched_processing: bool = False,
                    *method_args, **method_kwargs):
    """
    Use this method to execute another method in parallel. If you have a loop over an iterator, just replace the loop
    with a function (replace all continues with returns) and give the iterator to this method. For speed optimization,
    try to loop over your lists etc directly instead of looping over the range, then accessing the list elements via
    the looping index. I think that joblib otherwise needs to sync the whole list during each iteration, which can
    make it way more inefficient than sequential processing. Also also, try not to use open3d for in or outputs.
    A lot of open3d objects (including triangle meshes) are C++ wrappers than cannot be pickled, so you're getting an
    error here.
    Simple example usage:
        parallel_method(loop_function, iterator=my_iterator, num_jobs=8)
        -> this will call loop_function(iterator_el) with 8 parallel jobs,
           where iterator_el represents an element from the iterator my_iterator
    Advanced example usage:
        parallel_method(loop_function, iterator=my_iterator, iterator_name="iterator_el_name", num_jobs=8,
                        tqdm_message="Calling my function in parallel", other_method_arg=arg1, another_method_arg=arg2)
        -> this will call loop_function(iterator_el_name=iterator_el, other_method_arg=arg1, another_method_arg=arg2)
           with 8 parallel jobs, where iterator_el represents an element from the iterator my_iterator.
           Additionally, the TQDM progress bar will print the message "Calling my function in parallel"

    :param method: Callable method (do braces in the end!!)
    :param iterator: A list or any type of iterator
    :param iterator_name: What an element of the iterator should be called. Use this if the method has other arguments
                          and the iterator element is not the first one.
    :param start_from: A string to filter the given iterator, only starting from an element that matches this string.
                       Note that the iterator elements must be convertible to strings to allow for this filtering.
                       In case they are pathlib paths, this argument will be compared to the items' stems.
    :param stop_after: Similar to start_from argument, only does it define after which element in the iterator to stop.
    :param unpack_iterator: Choose this if you want to unpack the iterator, so if, e.g., your function expects two
                            arguments from two separate lists and you define your iterator as a zip of these two lists,
                            you should set this to true (note that this only makes sense if you don't provide an
                            iterator_name). You may provide an iterator as enumerate(zip(*some_lists)). The enumerate
                            is automatically recognized and, thus, additionally unpacked. If you only do
                            enumerate(some_list), you don't need to specify unpack_iterator=True (but it doesn't harm).
                            Don't try to accept a tuple for the enumerate iterator; it is always automatically unpacked
                            here.
    :param total: How many total iterations your iterator has -- some iterators, e.g., zip, may not allow to compute
                  the length, and it can also be expensive to compute it for some iterators, so if you can easily
                  compute it, provide it as an argument here. Without it, total is computed via len(iterator)
                  if possible. If you set verbose to False, this is not required.
    :param num_jobs: Number of jobs to execute in parallel. If <=1, the method is called in a simple loop.
    :param backend: str, never seen that it helps to change the default. Check joblib.parallel.py Parallel()
                    method for details.
    :param batched_processing: bool, default False. Instead of doing a normal parallelized iteration over the iterator,
                               we pre-separate the iterator into num_jobs batches, then we distribute the batches onto
                               separate threads and do a serialized execution on each. Could theoretically be faster
                               if one iteration is fast, but we have many iterations and some overhead.
                               Haven't seen any significant benefit in practice, though, yet.
    :param tqdm_message: Optional message to show during using TQDM progress bar.
    :param verbose: Whether to even print anything during the parallel execution of the method (True by default).
                    Note that tqdm_message is not shown when you set verbose to False even if you provide a
                    custom message.
    :param method_args: Optional additional positional method arguments.
                        Note that these come after the iterator element if iterator_name is not defined.
                        To avoid confusion, you should rather use method_kwargs.
    :param method_kwargs: Optional additional keyword method arguments.
    :return: List of outputs for each method call with the iterator as argument. If the method has no output,
             you'll get a list of Nones returned.
    """
    # If start_from and/or stop_after is defined, we shorten the iterator according to their choice.
    if start_from is not None or stop_after is not None:
        def is_str_convertable(my_obj):
            try:
                str(my_obj)
            except ValueError:
                return False
            return True

        iterator_update = []
        has_started = start_from is None
        for iterator_el in iterator:
            assert is_str_convertable(iterator_el), (f"The given iterator element {iterator_el} cannot be converted "
                                                     f"to a string, and thus the arguments start_from and stop_after "
                                                     f"do not work.")

            def check_sim(sim_str):
                return sim_str in (iterator_el.stem if isinstance(iterator_el, Path) else str(iterator_el))

            if start_from is not None:
                if check_sim(start_from):
                    has_started = True
                if not has_started:
                    continue
            iterator_update.append(iterator_el)
            if stop_after is not None and check_sim(stop_after):
                break
        iterator = iterator_update

    # Do you know how I could shorten this if else in python?
    def get_method_args(current_iterator_el):
        if iterator_name is None:
            # We give the option to the user to unpack the element of the iterator (useful for zip() iterator)
            if unpack_iterator:
                # We also unpack the enumerate iterator in addition to the normal unpacking (this is useful for the
                # combination of enumerate and zip; if only enumerate(some_list) or something is used, this doesn't
                # work, so we also check that the second element of the current iterator element is actually a tuple).
                # This might be a bit hard-coded, but I think it makes sense like this.
                if isinstance(iterator, enumerate) and isinstance(current_iterator_el[1], tuple):
                    return current_iterator_el[0], *current_iterator_el[1], *method_args
                else:
                    return *current_iterator_el, *method_args
            else:
                # We also unpack the enumerate even if unpack_iterator is False. Not sure if this is always the
                # desired behavior, but I find it more intuitive to have the arguments already separated in the
                # function calling, rather than having to unpack them within the function.
                if isinstance(iterator, enumerate):
                    return current_iterator_el[0], current_iterator_el[1], *method_args
                else:
                    return current_iterator_el, *method_args
        else:
            return method_args

    def get_method_kwargs(current_iterator_el):
        if iterator_name is None:
            return method_kwargs
        else:
            return {iterator_name: current_iterator_el, **method_kwargs}

    # If total is not provided, compute it as via the length of the iterator.
    # Since not all iterators support that, we need to catch the error.
    if total is None and verbose:
        try:
            total = len(iterator)
        except TypeError:
            total = None

    if isinstance(iterator, enumerate) and iterator_name is not None:
        warnings.warn(f"You provided a name for the iterator, but you use enumerate as an iterator, which means that "
                      f"you will have an argument called {iterator_name} that will get a tuple "
                      f"(index_number, actual_iterator_element). Ignore this warning if this is intentional.")

    if num_jobs > 1:
        # import locally, such that sequential processing of the method doesn't require joblib to be installed.
        from joblib import Parallel, delayed
        # Automatically split the iterator into num_jobs separate batches, then distribute the execution of the method
        # via another method that takes one of these batches and just loops over it.
        # Haven't really seen a big benefit of this, unfortunately.
        if batched_processing and total is not None:
            iterator_list = list(iterator)
            batch_size = len(iterator_list) // num_jobs
            iterator_batches = [iterator_list[i:min(i + batch_size, len(iterator_list))]
                                for i in range(0, len(iterator_list), batch_size)]

            def batch_process_method(iterator_batch, use_tqdm: bool = False):
                method_output = []
                inner_iterator = iterator_batch
                # Progress is only printed in one inner batch processing method.
                # It should reflect the overall progress, though.
                if use_tqdm:
                    inner_iterator = tqdm(iterator_batch, desc=f"{tqdm_message} Inner batch execution:")
                for iterator_el in inner_iterator:
                    method_output.append(method(*get_method_args(iterator_el), **get_method_kwargs(iterator_el)))
                # Return outputs as list here
                return method_output

            # We call the batch function without printing anything here.
            method_outputs = Parallel(n_jobs=num_jobs, backend=backend)(delayed(batch_process_method)(
                iterator_batch, use_tqdm=num_batch==0 and verbose)
                for num_batch, iterator_batch in enumerate(iterator_batches))
            # We have to serialize the outputs again, to avoid having a list with num_jobs lists as elements.
            return [method_output for method_outputs_ind in method_outputs for method_output in method_outputs_ind]
        else:

            def parallel_processing():
                return Parallel(n_jobs=num_jobs, backend=backend)(delayed(method)(
                    *get_method_args(iterator_el), **get_method_kwargs(iterator_el)) for iterator_el in iterator)

            if verbose:
                with tqdm_joblib(tqdm(desc=tqdm_message, total=total)):
                    return parallel_processing()
            else:
                return parallel_processing()
    else:
        iterator_loop = tqdm(iterator, desc=tqdm_message, total=total) if verbose else iterator
        method_output = []
        for iterator_el in iterator_loop:
            method_output.append(method(*get_method_args(iterator_el), **get_method_kwargs(iterator_el)))
        return method_output

def normalize_vector(vec_or_vecs):
    return np.asarray(vec_or_vecs) / np.maximum(np.linalg.norm(vec_or_vecs, axis=-1, keepdims=True), 1e-8)

def dot_vector_list(vector1, vector2, always_keep_dims=False) -> Union[float, np.ndarray]:
    """
    Given two vectors or lists of vectors, compute the dot product per vector. If one the two is a list of vectors,
    whereas the other one is a single vector, each vector element in the list is dot-multiplied with the same vector.
    @param vector1: Vector (d,) or list of vectors (n,d) (also supports more dimensions).
    @param vector2: Vector (d,) or list of vectors (n,d) (also supports more dimensions).
    @return: Scalar if vector1 and vector2 are single vectors, otherwise list of scalars (n,).
    """
    vector1, vector2 = np.asarray(vector1), np.asarray(vector2)
    original_dim = None
    if len(vector1.shape) > 2:
        original_dim = vector1.shape[:-1]
        vector1 = np.reshape(vector1, (-1, vector1.shape[-1]))
    if len(vector2.shape) > 2:
        if original_dim is None or len(vector2.shape[:-1]) > len(original_dim):
            original_dim = vector2.shape[:-1]
        vector2 = np.reshape(vector2, (-1, vector2.shape[-1]))
    if len(vector1.shape) == 1:
        vector1 = np.expand_dims(vector1, axis=0)
    if len(vector2.shape) == 1:
        vector2 = np.repeat(np.expand_dims(vector2, axis=0), repeats=vector1.shape[0], axis=0)
    if vector1.shape[0] == 1:
        vector1 = np.repeat(vector1, repeats=vector2.shape[0], axis=0)

    assert vector1.shape == vector2.shape
    dot_prod = np.einsum('ij, ij->i', vector1, vector2)
    if len(dot_prod) == 1 and not always_keep_dims:
        return dot_prod[0]
    if original_dim is not None:
        dot_prod = np.reshape(dot_prod, original_dim)
    return dot_prod

def get_axis_angle_rotation_between_vectors(
        a: Union[List, np.ndarray], b: Union[List, np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    Get the rotation in terms of axis and angle to align vector a with vector b.
    Note that the vectors should be non-zero and 3D. For the 2D case, you can
    simply compute the angle via the dot product between the normalized vectors; no axis needed.
    :param a: source vector to align with target vector (3D numpy array).
    :param b: target vector that source vector is aligned with (3D numpy array).
    :return: Tuple of axis (3D numpy vector) and angle (float, radian)
    """
    # Threshold to compensate numerical instability
    eps = 1e-8

    # cast to numpy array
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # norm and validity check
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < eps or norm_b < eps:
        raise ValueError("Input vectors must be non-zero.")

    a_normed = a / norm_a
    b_normed = b / norm_b

    cross_prod = np.cross(a_normed, b_normed)
    cross_norm = np.linalg.norm(cross_prod)
    dot_prod = np.clip(np.dot(a_normed, b_normed), -1.0, 1.0)

    # If vectors are nearly parallel or anti-parallel, cross is near zero
    if cross_norm < eps:
        if dot_prod > 0.0:
            # parallel: angle 0, axis arbitrary
            return np.array([1.0, 0.0, 0.0]), 0.0
        else:
            # anti-parallel: angle pi, pick any axis perpendicular to a
            # choose a helper vector not parallel to a
            helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = np.cross(a_normed, helper)
            axis /= np.linalg.norm(axis)
            return axis, np.pi

    axis = cross_prod / cross_norm
    # more stable than arccos near 0 and pi
    angle = np.arctan2(cross_norm, dot_prod)
    return axis, angle

def get_rotation_matrix_from_axis_and_angle(angle: float, axis: Union[List, np.ndarray, None] = None,
                                            homogeneous: bool = False) -> np.ndarray:
    """
    Returns a rotation matrix for 2D or 3D.

    - 2D: `axis` not required (is also ignored). Returns 2x2 (or 3x3 homogeneous).
    - 3D: `axis` must be a 3-vector. Returns 3x3 (or 4x4 homogeneous).

    Rotation is counter-clockwise in a right-handed coordinate system.
    For 2D, CCW is the usual mathematical convention.

    :param axis: 3D axis (shape (3,)) for 3D rotations. Ignored for 2D.
    :param angle: rotation angle in radians
    :param homogeneous: if True, returns homogeneous matrix (3x3 for 2D, 4x4 for 3D)
    :return: rotation matrix as np.ndarray
    """
    # Determine dimension from axis; if axis is None or not length 3, we treat it as 2D.
    if axis is None:
        dim = 2
    else:
        axis = np.asarray(axis).reshape(-1)
        dim = 3 if axis.size == 3 else 2

    if dim == 2:
        c = float(np.cos(angle))
        s = float(np.sin(angle))

        rot_mat = np.identity(3 if homogeneous else 2, dtype=float)
        rot_mat[:2, :2] = np.array([[c, -s], [s,  c]], dtype=float)

        return rot_mat

    # ---- 3D case ----
    if axis.size != 3:
        raise ValueError(f"3D rotation requires axis with 3 elements, got {axis.size}.")

    axis_normalized = normalize_vector(axis)
    a = float(np.cos(angle/2))
    b, c, d = -axis_normalized*np.sin(angle/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rot_mat = np.identity(4 if homogeneous else 3, dtype=float)
    rot_mat[:3, :3] = np.asarray([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                                  [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                                  [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]], dtype=float)
    return rot_mat

def get_translation_matrix(translate: np.ndarray) -> np.ndarray:
    """
    Get a (d+1)x(d+1) (homogeneous) translation matrix from a d-dimensional translation vector.
    :param translate: translation vector numpy array
    :return: (d+1)x(d+1) numpy matrix
    """
    translation_matrix = np.eye(len(translate)+1)
    translation_matrix[list(range(len(translate))), -1] = np.asarray(translate)
    return translation_matrix

def get_scaling_matrix(scale: Union[float, np.ndarray], homogeneous: bool = False, dim: int = 3) -> np.ndarray:
    """
    Get a scaling matrix from scaling scalar or vector.
    :param scale: scaling scalar or vector (if scalar, the same scaling is applied to each axis)
    :param homogeneous: Set True to get 4x4 homogeneous matrix.
    :param dim: Dimension we're working in. Default is 3, so you'll get a 3x3 or 4x4 (if homogeneous) matrix out.
    :return: dxd or (d+1)x(d+1) (if homogeneous is True) numpy array
    """
    scale_matrix = np.eye((dim+1) if homogeneous else dim)
    indices = np.arange(dim)
    scale_matrix[indices, indices] = scale
    return scale_matrix

def get_transformation_matrix(
        translate: Union[None, np.ndarray] = None, scale: Union[None, float, np.ndarray] = None,
        rot_axis: Union[np.ndarray, None] = None, rot_angle: Union[None, float] = None,
        dim: int = 3) -> np.ndarray:
    """
    Get a full (d+1)x(d+1) (homogeneous) transformation matrix from translation, scaling, and rotation.
    Note that we define the order as rotation @ scaling @ translation, so translation is applied first,
    then scaling, then rotation. Cf. get_translation_matrix(), get_scaling_matrix(),
    and get_rotation_matrix_from_axis_and_angle() for explanation of arguments.
    :param translate: translation vector numpy array (optional).
    :param scale: scaling scalar or vector numpy array (optional)
    :param rot_axis: rotation axis numpy array (optional)
    :param rot_angle: rotation angle numpy array (optional)
    :param dim: Dimension we're working in. Default is 3, so you'll get a 4x4 matrix out.
    :return (d+1)x(d+1) numpy matrix.
    """
    transformation_matrix = np.eye(dim + 1)
    if translate is not None:
        transformation_matrix = get_translation_matrix(translate) @ transformation_matrix
    if scale is not None:
        transformation_matrix = get_scaling_matrix(scale, homogeneous=True, dim=dim) @ transformation_matrix
    if rot_angle is not None:
        transformation_matrix = get_rotation_matrix_from_axis_and_angle(
            axis=rot_axis, angle=rot_angle, homogeneous=True) @ transformation_matrix
    return transformation_matrix

def matrix_contains_flip(matrix):
    return np.linalg.det(matrix) < 0

def get_flip_matrix(flip_axis=0, dim=3, homogeneous=True):
    """
    Returns the matrix that points can be multiplied with to be flipped along the provided axis.
    Basically, this is a scaling matrix with the flip_axis component being -1 and all others being 1.
    @param flip_axis: Axis around which to flip.
    @param dim: matrix dimension
    @param homogeneous: Whether a homogeneous matrix should be returned
                        (so if dim=3 and homogeneous, then a 4x4 matrix is returned)
    @return: Flipping matrix.
    """
    assert 0 <= flip_axis < dim
    matrix_dim = dim + 1 if homogeneous else dim
    flip_mat = np.eye(matrix_dim)
    flip_mat[flip_axis, flip_axis] = -1
    return flip_mat

def get_heatmap_gradient(values, reverse=False, colors=None):
    """
    Generates RGB values from normed input values in [0, 1].

    By default, this creates a 3-color heatmap:
        0.0 -> blue
        0.5 -> green
        1.0 -> red

    Custom gradients can be specified with `colors`, for example:
        colors=[[1, 0, 0], [1, 1, 1]]
        creates a red -> white gradient.

    Parameters
    ----------
    values : float or array-like
        Value(s) normed to [0, 1].
    reverse : bool, optional
        If True, reverses the direction of the gradient.
    colors : array-like, optional
        Sequence of RGB colors. Each color must have three components.
        Values are expected to be in [0, 1].

    Returns
    -------
    np.ndarray
        RGB color for a single value, or an array of RGB colors.
    """
    values = np.array(values)

    if colors is None:
        colors = np.asarray([
            [0.0, 0.0, 1.0],  # blue
            [0.0, 1.0, 0.0],  # green
            [1.0, 0.0, 0.0],  # red
        ])
    else:
        colors = np.asarray(colors, dtype=float)

    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("`colors` must be an array-like object with shape (n_colors, 3).")

    if len(colors) < 2:
        raise ValueError("At least two colors are required to compute a gradient.")

    if reverse:
        values = 1 - values

    # If values is a single number, expand it here, but collapse again before returning.
    collapse = False
    if len(values.shape) == 0:
        values = np.expand_dims(values, axis=0)
        collapse = True

    values = values.astype(float)

    # Optional safety: ensure values stay inside the valid range.
    values = np.clip(values, 0.0, 1.0)

    # Scale values into color-segment coordinates.
    # Example with 3 colors:
    #   value 0.00 -> segment 0, t=0.0
    #   value 0.25 -> segment 0, t=0.5
    #   value 0.50 -> segment 1, t=0.0
    #   value 1.00 -> segment 1, t=1.0
    n_segments = len(colors) - 1
    scaled = values * n_segments

    segment_indices = np.floor(scaled).astype(int)
    segment_indices = np.clip(segment_indices, 0, n_segments - 1)

    t = scaled - segment_indices
    t = t[:, np.newaxis]

    start_colors = colors[segment_indices]
    end_colors = colors[segment_indices + 1]

    rgb = start_colors * (1 - t) + end_colors * t

    return rgb[0] if collapse else rgb

def get_error_heatmap_colors(errors: np.ndarray, min_value: float = None, max_value: float = None,
                             reverse: bool = False, heatmap_colors=None):
    """
    This method returns colors corresponding to the provided errors, red for high, blue for low, green for middle.
    The user can also choose to invert them and choose their own min and max values, in which case
    the errors are clipped to this range.
    :param errors: 1D Numpy array of error values.
    :param min_value: Minimum value (float) of the error, if not provided np.min(errors) is used.
    :param max_value: Maximum value (float) of the error, if not provided np.max(errors) is used.
    :param reverse: Set True to invert the color gradient, so to have red for low, and blue for high.
    :param heatmap_colors: Sequence of colors to represent error transition, cf. get_heatmap_gradient().
    :return Numpy array of heatmap RGB [0,1] colors.
    """
    # Defines the max and min of the heatmap range. If max_value is not none it is clipped between max_value.
    # Otherwise defines max and min as the min and max elements of error.
    min_value = np.min(errors) if min_value is None else min_value
    max_value = np.max(errors) if max_value is None else max_value
    if not min_value < max_value:
        raise AssertionError("min_value must be strictly smaller than max_value for the heatmap color "
                             "range to work.git add.")
    errors_for_colors = (errors.clip(min_value, max_value) - min_value) / (max_value - min_value)
    return get_heatmap_gradient(errors_for_colors, reverse=reverse, colors=heatmap_colors)

def is_existing_path(path):
    return isinstance(path, (str, Path)) and Path(path).exists()

def get_all_files_in_path(
        path: Union[List[Union[str, Path]], Union[str, Path]], file_suffixes: Union[str, List[str], None] = None,
        nested: bool = True, regex_names: Union[str, List[str], None] = "*",
        exclude_regex_names: Union[str, List[str], None] = None,
        remove_suffix_duplicates: bool = False, priority_suffix_list: Union[str, List[str], None] = None,
        added_since: Union[str, None] = None, added_before: Union[str, None] = None) -> List[Path]:
    """
    Return all paths within a directory, filtered by optional suffixes and names.
    The returned list is sorted (via os_sorted from natsort if available, otw normal sorted() function).
    @param path: Path to directory, or list of such paths. If it's a file,
    @param file_suffixes: str or list of str specifying the file types/suffixes to get from the provided path(s).
                          E.g. input for meshes: [".stl", ".ply", ".obj"].
                          The suffix can also be specified via regex_names.
    @param nested: Whether to consider subdirectories as well. Default is True.
    @param regex_names: You can narrow down a list for allowed files by supplying a regex for the file name.
                        E.g., you may know the name of a file, but not its extension, so you supply the name
                        here and the function returns all files of that name that fit the suffixes.
                        Use "*" for arbitrary names, e.g., "0001_*" will return all files whose names
                        start with "0001_".
    @param exclude_regex_names: Files that match this regex will be excluded from the list.
                                This overwrites regex_names, so if you provide regex_names="0001_*",
                                and exclude_regex_names="0001_23*", only those files that start with "0001_"
                                will be returned that don't start with "0001_23".
    @param remove_suffix_duplicates: If there are multiple files with the same stem, differing only in the suffix,
                                     only return one of them; priority can be specified in priority_suffix_list.
    @param priority_suffix_list: If suffix duplicates are chosen to be removed, define via list of strings the order
                                 of priority, e.g., [".obj", ".ply", ".stl"] to always keep .obj over .ply and .stl
                                 files (the period at the beginning is required).
    @param added_since: Optionally only return files added since the date you provide here (YYYY-MM-DD).
    @param added_before: Optionally only return files added before the date you provide here (YYYY-MM-DD).
    @return: Sorted list of paths that fulfill the criteria of belonging to the provided suffixes
             and are within the given path parameters. The list may be empty if none such file exists.
    """
    # If multiple paths are provided, we search through each and streamline the output
    if isinstance(path, list):
        return [file for path_ind in path for file in
                get_all_files_in_path(
                    path_ind, file_suffixes=file_suffixes, nested=nested, regex_names=regex_names,
                    exclude_regex_names=exclude_regex_names, remove_suffix_duplicates=remove_suffix_duplicates,
                    priority_suffix_list=priority_suffix_list, added_before=added_before, added_since=added_since)]

    # Prepare arguments for processing
    pathlib_path = Path(path)

    if not isinstance(regex_names, (str, list)) or len(regex_names) == 0:
        regex_names = ["*"]
    elif isinstance(regex_names, str):
        regex_names = [regex_names]

    if isinstance(exclude_regex_names, str):
        exclude_regex_names = [exclude_regex_names]

    if file_suffixes is None:
        file_suffixes = []
    elif isinstance(file_suffixes, str):
        file_suffixes = [file_suffixes]
    file_suffixes = [suffix if suffix.startswith(".") else f".{suffix}" for suffix in file_suffixes]

    # Compile the regexes.
    def glob_to_regex(pattern: str):
        """Convert a shell-style glob pattern to a regular expression."""
        # Escape all regex special characters except for * and ?
        escaped = re.escape(pattern).replace(r'\*', '.*').replace(r'\?', '.')
        # Anchors for start and end of string
        return '^' + escaped + '$'

    def get_compiled_regexes(provided_regexes) -> List:
        regexes = []
        for regex in provided_regexes:
            if len(file_suffixes) == 0 or any([len(regex) > len(suffix) and regex[-len(suffix):] == suffix for suffix in file_suffixes]):
                regexes.append(re.compile(glob_to_regex(regex)))
            else:
                for suffix in file_suffixes:
                    regexes.append(re.compile(glob_to_regex(regex + suffix)))
        return regexes

    allowed_compiled_regexes = get_compiled_regexes(regex_names)
    excluded_compiled_regexes = get_compiled_regexes(exclude_regex_names) if exclude_regex_names is not None else []

    if added_before is not None:
        added_before = datetime.strptime(added_before, '%Y-%m-%d')
    if added_since is not None:
        added_since = datetime.strptime(added_since, '%Y-%m-%d')

    # We basically look for all files in the provided directory and check if
    # that file matches any of the provided regexes and doesn't match any of the excluded regexes.
    # Also, if added_since and/or added_before were provided, we filter with those as well.
    def is_matching_file(file_path: Path):
        if not file_path.is_file():
            return False
        if len(excluded_compiled_regexes) > 0 and any([
            exclude_regex.match(file_path.name) for exclude_regex in excluded_compiled_regexes]):
            return False
        file_allowed = any([allowed_regex.match(file_path.name) for allowed_regex in allowed_compiled_regexes])
        if added_since is not None:
            file_allowed = file_allowed and datetime.fromtimestamp(file_path.stat().st_mtime) >= added_since
        if added_before is not None:
            file_allowed = file_allowed and datetime.fromtimestamp(file_path.stat().st_mtime) < added_before
        return file_allowed

    # If we get a file instead of a directory, we check if it matches the supplied filtering.
    # If it doesn't, we print a warning, since this seems unintentional.
    if not pathlib_path.is_dir():
        if is_matching_file(pathlib_path):
            return [pathlib_path]
        else:
            warnings.warn(f"{pathlib_path} is not a directory, but it's also not a file that matches with your "
                          f"specified regexes and suffixes. Is this intentional?")
            return []

    # Sort the paths
    try:
        from natsort import os_sorted
    except ImportError:
        os_sorted = sorted
    path_list = os_sorted([path for path in pathlib_path.glob("**/*" if nested else "*") if is_matching_file(path)])

    # Potentially remove file duplicates.
    if remove_suffix_duplicates:
        path_list_updated = []

        def get_path_priority(current_path: Path):
            if priority_suffix_list is None or len(priority_suffix_list) == 0:
                return 0
            if current_path.suffix in priority_suffix_list:
                return priority_suffix_list.index(current_path.suffix)
            return len(priority_suffix_list)

        for path in path_list:
            if len(path_list_updated) == 0 or path.stem != path_list_updated[-1].stem:
                path_list_updated.append(path)
            elif path.stem == path_list_updated[-1].stem:
                if get_path_priority(path) < get_path_priority(path_list_updated[-1]):
                    path_list_updated[-1] = path
        path_list = path_list_updated

    return path_list


class Plane:
    """
    A Plane is defined by a point on the plane (the anchor) and a normal vector that's perpendicular to the plane
    (the direction). This class saves these attributes and offers additional methods to build a plane from various
    inputs, intersect planes, also work with lines (cf. Line class below), and meshes.
    """
    point_on_plane: Union[list, np.ndarray]
    plane_normal: Union[list, np.ndarray]
    half_plane_direction: Union[list, np.ndarray] = None
    is_half_plane: bool = False


    def __init__(self, point_on_plane: Union[list, np.ndarray], plane_normal: Union[list, np.ndarray],
                 half_plane_direction: Union[list, np.ndarray] = None):
        """
        :param point_on_plane: Any point on the plane. Shape (3).
        :param plane_normal: Vector perpendicular to plane (shape (3), no need to normalize it before;
                             it's normalized in the __init__ method).
        """
        self.set_point(point_on_plane)
        self.set_normal(plane_normal)
        if half_plane_direction is not None:
            self.set_half_plane_direction(half_plane_direction)

    @staticmethod
    def get_plane_from_three_points(
            first_point: Union[list, np.ndarray], second_point: Union[list, np.ndarray],
            third_point: Union[list, np.ndarray]) -> "Plane":
        plane_anchor = np.mean(np.asarray([first_point, second_point, third_point]), axis=0)
        plane_normal = np.cross(second_point - first_point, third_point - first_point)
        return Plane(point_on_plane=plane_anchor, plane_normal=plane_normal)

    @staticmethod
    def get_plane_from_two_points_and_one_vector(
            first_point: Union[list, np.ndarray], second_point: Union[list, np.ndarray],
            vector_on_plane: Union[list, np.ndarray]) -> "Plane":
        plane_anchor = np.mean(np.asarray([first_point, second_point]), axis=0)
        plane_normal = np.cross(second_point - first_point, vector_on_plane)
        return Plane(point_on_plane=plane_anchor, plane_normal=plane_normal)

    @staticmethod
    def get_plane_from_one_point_and_two_vectors(
            point_on_plane: Union[list, np.ndarray], first_vector: Union[list, np.ndarray],
            second_vector: Union[list, np.ndarray]) -> "Plane":
        return Plane(point_on_plane=point_on_plane,
                     plane_normal=np.cross(first_vector, second_vector))


    def get_point(self) -> np.ndarray:
        """
        :return: np array representing a point on the plane (the one the Plane instance was initialized with)
        """
        return np.array(self.point_on_plane)

    def set_point(self, point_on_plane: Union[list, np.ndarray]):
        """
        Setter method for plane anchor.
        :param point_on_plane: Any point on the plane. Shape (3).
        :return: None
        """
        self.point_on_plane = np.array(point_on_plane)

    def get_normal(self) -> np.ndarray:
        """
        :return: plane's normal vector as np array, normalized to Euclidean length 1
        """
        return np.array(self.plane_normal)

    def set_normal(self, plane_normal: Union[list, np.ndarray]):
        """
        Setter method for plane direction.
        :param plane_normal: Vector perpendicular to plane (shape (3), no need to normalize it before).
        :return: None
        """
        self.plane_normal = np.asarray(plane_normal)
        self.plane_normal = self.plane_normal / np.linalg.norm(self.plane_normal)

    def flip_normal(self):
        self.set_normal(-self.plane_normal)

    def set_half_plane_direction(self, half_plane_direction: Union[list, np.ndarray]):
        self.half_plane_direction = normalize_vector(np.asarray(half_plane_direction))
        if not np.dot(self.plane_normal, self.half_plane_direction) < 1e-5:
            raise AssertionError("Given half plane direction is not perpendicular to plane normal")
        # make sure the half plane direction really lies inside the plane
        self.half_plane_direction = normalize_vector(self.project_vectors_to_plane(half_plane_direction))
        self.is_half_plane = True

    def get_half_plane_direction(self):
        return self.half_plane_direction

    def get_type(self) -> str:
        if self.is_half_plane:
            return "half-plane"
        return "plane"

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: Tuple(plane anchor, plane normal). The order of object instantiation is kept,
        so you can input these params in that order when creating a new Plane instance.
        """
        return self.get_point(), self.get_normal()

    def get_mesh(self, size: float = 1, other_centroid: Union[list, np.ndarray] = None):
        """
        Create a plane mesh using the Plane's point_on_plane as centroid unless another is provided.
        :param size: float that defines the half length of the plane's side
        :param other_centroid: Optionally provide another centroid if you want to avoid having a shifted plane
                               (warning is printed if that point isn't on the plane)
        :return: plane mesh (four corner vertices connected by two triangles)
        """
        from src.objects.mesh import Mesh

        if self.is_half_plane:
            u = self.half_plane_direction
        else:
            # Compute two orthogonal vectors in the plane
            if abs(self.plane_normal.dot([0, 0, 1])) < 1:
                u = np.cross(self.plane_normal, [0, 0, 1])
            else:
                u = np.asarray([1, 0, 0])
        v = np.cross(self.plane_normal, u)
        u = u / np.linalg.norm(u) * size
        v = v / np.linalg.norm(v) * size

        if other_centroid is None:
            plane_centroid = self.point_on_plane
        else:
            plane_centroid = np.ndarray(other_centroid)
            if self.compute_distance_points_to_plane(plane_centroid) > 1e-5:
                warnings.warn("You provided another plane centroid, but the point is actually not on the plane."
                              "Define another plane instance that covers this point to remove this warning.")

        if self.is_half_plane:
            # A half plane has a defined boundary line on one side,
            # so we shift the centroid, such that the plane mesh respects that boundary.
            plane_centroid = plane_centroid + u

        # Compute the four corners of the plane
        corners = np.array([
            plane_centroid + u + v,
            plane_centroid + u - v,
            plane_centroid - u - v,
            plane_centroid - u + v
        ])

        return Mesh(vertices=corners, triangles=np.asarray([[0, 1, 2], [2, 3, 0]]))

    @staticmethod
    def fit_plane_to_points(points: Union[List, np.ndarray]) -> "Plane":
        """
        Given a set of points, compute a plane that fits the points as closely as possible
        (minimizing squared Euclidean distance).
        In the normal case, the normal of the plane is in a right-hand system with the vectors between
        first and second projected point towards all points' center. This only doesn't apply in edge cases
        where we get linear dependence between these directions, in which case by the point causing the edge
        case is replaced by the next point in order.
        :param points: List or numpy array of 3D points.
        :return: point center (representing a point on the plane, 3D np array), plane normal (normalized, 3D np array)
        """
        # check input
        points = np.asarray(points)
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise TypeError(f"Given points are not a 3D point array: {points}")
        if not len(points) >= 3:
            raise ValueError("You must provide at least three points to fit a plane through.")

        # Compute the centroid of the points
        centroid = np.mean(points, axis=0)

        # Subtract the centroid from the points
        centered_points = points - centroid

        # Compute the covariance matrix
        cov_matrix = np.dot(centered_points.T, centered_points) / centered_points.shape[0]

        # Perform SVD -> The best plane is computed via the vector that minimizes the variance in the points
        # (thus is perpendicular to the plane that has the largest variance in points)
        try:
            _, _, vh = np.linalg.svd(np.asarray(cov_matrix, dtype=float))
        except np.linalg.LinAlgError:
            raise ValueError("The points you provided are all on the same line.")

        # The normal to the plane is the last column of vh (corresponding to the smallest singular value)
        # By construction, this is alreay normalized
        normal = vh[-1]

        initial_plane = Plane(centroid, normal)

        # We adjust the direction of the normal to make it deterministic:
        # We find two distinct directions on the plane and compute their cross product.
        # To find these two directions, we go through the given points in order
        # and compute each point's projected direction to the center.
        # We take the first two directions that are linearly independent.
        for num_point, point in enumerate(points):
            # project first point to plane
            projected_point = initial_plane.project_points(point)
            # We cannot compute the direction to the center if the point's projection falls
            # directly on the center, so we skip it in that case.
            if not np.array_equal(projected_point, centroid):
                # Compute the the difference between center and projected
                # point to give us the first direction in the plane.
                first_direction = normalize_vector(centroid - projected_point)
                # Iterate through remaining points to find the second direction
                for next_point in points[num_point + 1:]:
                    next_projected_point = initial_plane.project_points(next_point)
                    # The second valid point's projection cannot fall on the center of the first point's direction.
                    if (not np.array_equal(next_projected_point, projected_point) and not
                    np.array_equal(next_projected_point, centroid)):
                        second_direction = normalize_vector(centroid - next_projected_point)
                        # The second direction also cannot be linearly dependent on the first direction.
                        # It could be that the two points lie exactly on opposite sides of the center,
                        # in which case their projections would not equal in any way, but the directions
                        # are still not enough to get the normal out.
                        # We check for linear dependence via the dot product with some
                        # threshold for safety/numerical stability.
                        if abs(np.dot(first_direction, second_direction)) < 0.999:
                            # Compute new normal. We sanity-check that it equals the old normal, except for the sign.
                            adjusted_normal = normalize_vector(np.cross(first_direction, second_direction))
                            if abs(np.dot(normal, adjusted_normal)) < 0.999:
                                raise AssertionError("You didn't only change the sign of the normal. Something's off.")
                            return Plane(centroid, adjusted_normal)

        raise AssertionError("I don't think it's possible to reach this point here. Check why it still happened.")

    def compute_distance_points_to_plane(self, points_to_compute_dist_for: Union[List, np.ndarray], compute_absolute: bool = True) -> Union[np.ndarray, float]:
        """
        Project a set of points onto a plane, represented via one point on the plane and the normal.
        :param points_to_compute_dist_for: The input points that the distance to the plane should be computed for;
                                           can also be a single point. Shape (3) or (nx3)
        :param compute_absolute: Set to False to also allow negative distances.
        :return: np array of distances, one for each given point (it's a simple float if only one point is given)
        """
        # Vector from plane_point to the point
        points_to_plane_points = np.asarray(points_to_compute_dist_for) - self.point_on_plane

        # Distance equals the projection of this connecting vector onto the normal
        distances = np.dot(points_to_plane_points, self.plane_normal)

        if compute_absolute:
            distances = np.abs(distances)

        return distances

    def compute_distance_to_other_plane(self, other_plane: "Plane", compute_absolute: bool = True) -> float:
        """
        Compute distance from this plane to another plane. This doesn't consider half planes currently.
        :param other_plane: The other plane (type Plane) to compute distance to.
        :param compute_absolute: Set this to False to also allow negative distances (this plane's normal defines sign)
        :return: Float representing the distance to the other plane (optionally with sign).
        """
        other_plane_point = other_plane.get_point()
        return self.compute_distance_points_to_plane(other_plane_point, compute_absolute=compute_absolute)

    def project_points(self, points_to_project: np.ndarray) -> np.ndarray:
        """
        Project a set of points onto a plane, represented via one point on the plane and the normal.
        Half-plane structure is not respected yet.
        :param points_to_project: The input points that are to be projected onto the plane; can also be a single point.
                                  Shape (3) or (nx3)
        :return np array of points projected onto plane. Shape same as input, so (3) or (nx3).
        """
        if self.is_half_plane:
            raise NotImplementedError("One could do something similar as for the line here.")
        # Projection of the vector onto the normal
        normal_dot = self.compute_distance_points_to_plane(points_to_project, compute_absolute=False)

        # Distinguish between case where points_to_project is only a single point vs multiple
        if points_to_project.ndim == 2:
            normal_dot = np.expand_dims(normal_dot, axis=-1)
        projections_onto_normal = normal_dot * self.plane_normal

        # Subtract this projection from the original point to get the projection on the plane
        return points_to_project - projections_onto_normal

    def align_points_along_plane(self, points_to_align: np.ndarray) -> np.ndarray:
        """
        This method doesn't simply project the points to the plane. Instead, it aligns the points along the plane.
        This means that if you put a plane through the aligned points, the resulting plane will be parallel to this
        Plane object. The method to get these aligned points works like this:
        1) project points to plane
        2) compute average signed distance of points to plane
        3) To each projected point from 1), add the normal times the average signed distance from 2).
        :param points_to_align: Numpy array of 3D points (nx3).
        :return: numpy array of same dim, containing aligned points.
        """
        points_projected = self.project_points(points_to_align)
        point_distances_signed = self.compute_distance_points_to_plane(points_to_align, compute_absolute=False)
        avg_distance = np.mean(point_distances_signed)
        return points_projected + self.plane_normal * avg_distance

    def project_vectors_to_plane(self, vectors_to_project: np.ndarray) -> np.ndarray:
        """
        Project a set of vectors onto a plane defined by its normal vector.
        Half-plane structure is not taken into account here, as vectors can be moved in space.
        :param vectors_to_project: The input vectors that are to be projected onto the plane; can also be a single vector.
                                   Shape (3) or (nx3)
        :return: np array of vectors projected onto the plane. Shape same as input, so (3) or (nx3).
                 The vectors are not normalized.
        """
        # Projection of the vector onto the normal
        normal_dot = np.dot(vectors_to_project, self.plane_normal)
        # Distinguish between case where vectors_to_project is only a single vector vs multiple
        if vectors_to_project.ndim == 2:
            normal_dot = np.expand_dims(normal_dot, axis=-1)
        projections_onto_normal = normal_dot * self.plane_normal

        # Subtract this projection from the original vector to get the projection on the plane
        return vectors_to_project - projections_onto_normal


    def check_side_of_plane(self, points_to_check_side_for: np.ndarray) -> np.ndarray:
        """
        Given one or multiple points, compute the side of the provided plane the points are on w.r.t.
        the plane's normal direction. Half-plane not considered here.
        :param points_to_check_side_for: One or multiple 3D points.
        :return: Numpy array with 1 and -1 entry for each provided point indicating positive and negative side on the plane.
        """
        # Determine the side based on the sign of the dot product
        return np.sign(self.compute_distance_points_to_plane(points_to_check_side_for, compute_absolute=False))

    def intersect_plane(self, other_plane: "Plane") -> "Line":
        """
        cf. docs of intersect_planes
        """
        return self.intersect_planes(self, other_plane)


    @staticmethod
    def intersect_planes(plane1: "Plane", plane2: "Plane") -> "Line":
        """
        Compute the line that intersects the given two planes.
        :param plane1: First plane (to be intersected with plane2)
        :param plane2: Second plane (to be intersected with plane1)
        :return: instance of Line that represents the intersection of the two planes
        :raise: ValueError if the two planes are parallel.
        """
        if plane1.is_half_plane or plane2.is_half_plane:
            raise NotImplementedError("Half-plane intersection is not implemented yet.")
        plane1_point, plane1_normal = plane1.get_point(), plane1.get_normal()
        plane2_point, plane2_normal = plane2.get_point(), plane2.get_normal()

        # Normalize the plane normals
        n1, n2 = normalize_vector([plane1_normal, plane2_normal])

        # Check if the planes are parallel by calculating the cross product of the normals
        cross_n1_n2 = np.cross(n1, n2)
        if np.allclose(cross_n1_n2, 0):
            # Planes are parallel
            raise ValueError("The two planes are parallel, so there's either no intersection or the planes are the same.")

        # The direction of the intersecting line is given by the cross product of the normals
        direction = normalize_vector(cross_n1_n2)

        # To find a point on the intersecting line, we solve for a specific point.
        # We can set one of the coordinates to zero and solve the resulting system.
        A = np.array([n1, n2])
        d = np.array([np.dot(n1, plane1_point), np.dot(n2, plane2_point)])

        # Find a point on the intersection line by setting one coordinate to zero and solving
        if not np.allclose(direction[0], 0):
            A_reduced = A[:, 1:]
            point_reduced = np.linalg.solve(A_reduced, d)
            point_on_line = np.array([0, *point_reduced])
        elif not np.allclose(direction[1], 0):
            A_reduced = A[:, [0, 2]]
            point_reduced = np.linalg.solve(A_reduced, d)
            point_on_line = np.array([point_reduced[0], 0, point_reduced[1]])
        else:
            A_reduced = A[:, :2]
            point_reduced = np.linalg.solve(A_reduced, d)
            point_on_line = np.array([*point_reduced, 0])

        # return the line parametrization
        return Line(point_on_line, direction)

    def slice_mesh(self, mesh: "Mesh", flip_direction: bool = False, cap: bool = True) -> "Mesh":
        """
        Slice mesh through plane and only keep vertices that are above the plane, so plane direction matters.
        Main computation is outsourced to trimesh.intersections.slice_mesh_plane()
        :param mesh: Triangle mesh of type Mesh
        :param flip_direction: If you notice that the direction is wrong, you can flip it by setting this arg to True.
        :param cap: Whether to close the mesh at the plane intersection with flat triangles.
        :return: Sliced mesh of type Mesh.
        """
        import trimesh
        normal = np.array(self.plane_normal)
        if flip_direction:
            normal = -normal

        mesh_trimesh = mesh.get_trimesh_mesh()
        try:
            sliced_mesh = trimesh.intersections.slice_mesh_plane(
                mesh_trimesh, plane_origin=self.point_on_plane, plane_normal=normal, cap=cap)
        except ValueError as e:
            if cap:
                raise ValueError("It could be that you're using an old trimesh version, which required the input mesh"
                                 f"to be watertight. Upgrade to 3.16.0 or later to avoid this. Original error: {e}")
            else:
                raise ValueError(e)
        return mesh.get_from_trimesh_mesh(sliced_mesh)

        #mesh_open3d = mesh.get_open3d_mesh(use_legacy=True)
        #sliced_mesh = mesh_open3d.clip_plane(point=self.point_on_plane.tolist(), normal=normal.tolist())
        #return Mesh.get_from_open3d_mesh(sliced_mesh)

    def intersect_mesh(self, mesh: "Mesh") -> List[np.ndarray]:
        """
        Intersect plane with a mesh, resulting in a set of line segments (each segment is one triangle of the
        mesh that is intersected by the plane). Note that there's no order, so it's not given that the second
        line segment's start point equals the first line segment's end point!
        Main computation is outsourced to open3d's slice_mesh()
        :param mesh: Triangle mesh of type Mesh
        :return: line segments (List of 3D point 2-pairs)
        """
        mesh_open3d = mesh.get_open3d_mesh(use_legacy=True)
        line_segments_open3d = mesh_open3d.slice_plane(
            point=self.point_on_plane.tolist(), normal=self.plane_normal.tolist())
        line_segment_positions = line_segments_open3d.point.positions.numpy()
        line_segment_indices = line_segments_open3d.line.indices.numpy()
        line_segments = [line_segment_positions[index_pair] for index_pair in line_segment_indices]
        return line_segments


class Line:
    point_on_line: Union[list, np.ndarray]
    line_direction: Union[list, np.ndarray]
    is_ray: bool

    def __init__(self, point_on_line: Union[list, np.ndarray], line_direction: Union[list, np.ndarray],
                 is_ray: bool = False):
        """
        :param point_on_line: Any point on the line. Shape (3).
        :param line_direction: Vector that goes along the line (shape (3), no need to normalize it before;
                               it's normalized in the __init__ method).
        :param is_ray: If you want to treat the line instead as a ray (so only going into positive direction
                       from point_on_line), set this to True. Class methods' behavior is adjusted accordingly, e.g.,
                       points behind the ray will get np.nan values assigned for the projection,
                       and the distance also takes the ray's end point into account.
        """
        self.set_point(point_on_line)
        self.set_direction(line_direction)
        self.is_ray = is_ray

    def get_point(self) -> np.ndarray:
        """
        :return: np array representing a point on the line (the one the line instance was initialized with)
        """
        return self.point_on_line

    def set_point(self, point_on_line: Union[list, np.ndarray]):
        """
        Setter method for line anchor.
        :param point_on_line: Any point on the line. Shape (3).
        :return: None
        """
        self.point_on_line = np.asarray(point_on_line)

    def get_direction(self) -> np.ndarray:
        """
        :return: line's directional normal vector as np array, normalized to Euclidean length 1
        """
        return self.line_direction

    def set_direction(self, line_direction: Union[list, np.ndarray]):
        """
        Setter method for line direction.
        :param line_direction: Vector that goes along the line (shape (3), no need to normalize it before).
        :return: None
        """
        self.line_direction = np.asarray(line_direction)
        self.line_direction = self.line_direction / np.linalg.norm(self.line_direction)

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: Tuple(line anchor, line direction). The order of object instantiation is kept,
        so you can input these params in that order when creating a new Line instance.
        """
        return self.get_point(), self.get_direction()

    def get_type(self) -> str:
        if self.is_ray:
            return "ray"
        return "line"

    def get_mesh(self, size: float = 1, radius: float = 0.15, other_centroid: Union[list, np.ndarray] = None):
        """
        Create a line mesh using the line's point_on_line as center point unless another is provided.
        Rays are taken into account, unless you mess it up with giving another center point.
        :param size: float that defines half of the length of the line mesh
                     (size goes in both directions from center point)
        :param radius: float that defines the radius of the cylindrical line mesh
        :param other_centroid: Optionally provide another center point if you want to avoid having a shifted line
                             (warning is printed if that point isn't on the line/ray)
        :return: line mesh (cylinder)
        """
        raise NotImplementedError("Due to a lot of dependencies, this method is currently unavailable.")


    @staticmethod
    def fit_line_to_points(points: Union[List, np.ndarray]) -> "Line":
        """
        Given a set of points, compute a line that fits the points as closely as possible
        (minimizing squared Euclidean distance).
        In the normal case, the direction of the line points from the first point towards all points' center.
        This only doesn't apply if the first point is directly on the center, in which case the next point
        decides the sign of the direction.
        :param points: List or numpy array of 3D points.
        :return: Line instance
        """

        # check input
        points = np.asarray(points)
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise TypeError(f"Given points are not a 3D point array: {points}")
        if not len(points) >= 3:
            raise ValueError("You must provide at least three points to fit a plane through.")

        # Compute the centroid of the points
        centroid = np.mean(points, axis=0)

        # Subtract the centroid from the points
        centered_points = points - centroid

        # Compute the covariance matrix
        cov_matrix = np.dot(centered_points.T, centered_points)

        # Perform SVD -> line is computed via the vector that represents the largest variance in the data
        # (first principal component).
        try:
            _, _, vh = np.linalg.svd(np.asarray(cov_matrix, dtype=float))
        except np.linalg.LinAlgError:
            raise ValueError("The points you provided are all on the same point.")

        # The direction of the line is the first column of vh (corresponding to the largest singular value)
        direction = vh[0]

        initial_line = Line(centroid, direction)

        # We update the sign of the direction by projecting the points (in order) onto the line,
        # and the first projection that doesn't fall directly on the center, then defines the sign
        # of the direction -- we subtract the projection from the center.
        # So in the normal case, the direction of the line goes from the first point towards all points' center.
        for point in points:
            projected_point = initial_line.project_points(point)
            if not np.array_equal(projected_point, centroid):
                adjusted_direction = normalize_vector(centroid - projected_point)
                if abs(np.dot(direction, adjusted_direction)) < 0.999:
                    raise AssertionError("Updated line direction is not in line with the original direction. "
                                         "We should only get at most a sign change, but nothing else. Check.")
                return Line(centroid, adjusted_direction, is_ray=False)

        raise AssertionError("Every of the individual points projected to the fitted line equals the center "
                             "of the points. This shouldn't happen unless all points are the same, "
                             "in which case the SVD should have already thrown an error, so this error "
                             "here should never happen. Check why it did. ^^")

    def compute_points_distance_to_line(self, points_to_compute_dist_for: np.ndarray) -> np.ndarray:
        """
        Compute the distance on a line that a set of points have (how far away are the points from the line).
        This method differs from compute_points_distance_on_line() in that it doesn't project the points onto the line
        and then computes their distance to the anchor, but rather it computes the points' actual distance to the line
        (measured orthogonally to the line).
        The ray property is considered here: Points that lie behind the ray have as closest point on the ray the anchor,
        so their distance to the ray is that between the point and the anchor.
        :param points_to_compute_dist_for: The input points that the distance on the line should be computed for;
                                           can also be a single point. Shape (3) or (nx3)
        :return np array of distances. Single float if single point of shape (3) is given, otherwise n-dim vector.
        """

        # We project the points onto the line, then compute the distance
        # between the input points and the projected points.
        projected_points = self.project_points(points_to_compute_dist_for)

        # We replace nan values with the ray anchor
        if projected_points.ndim == 2:
            projected_points[np.isnan(projected_points[:, 0])] = self.point_on_line
        else:
            if np.isnan(projected_points[0]):
                projected_points = np.array(self.point_on_line)

        return np.linalg.norm(projected_points - points_to_compute_dist_for, axis=-1)

    def compute_points_distance_on_line(self, points_to_compute_dist_for: np.ndarray) -> np.ndarray:
        """
        Compute the distance on a line that a set of points have (how far along the line from the line's anchor).
        This method differs from compute_points_distance_to_line() in that it projects the points onto the line and
        then computes their distance to the line's anchor, rather than measuring the points' actual distance
        to the line itself.
        Note that the ray property is not considered here (you can just take the negative numbers --
        these would not lie on the ray anymore.
        :param points_to_compute_dist_for: The input points that the distance on the line should be computed for;
                                           can also be a single point. Shape (3) or (nx3)
        :return np array of distances. Single float if single point of shape (3) is given, otherwise n-dim vector.
        """
        # Vector from point_on_line to the points to project
        points_to_line_points = points_to_compute_dist_for - self.point_on_line

        # Projection of the vector onto the line direction
        direction_dot = np.dot(points_to_line_points, self.line_direction)

        return direction_dot

    def project_points(self, points_to_project: np.ndarray) -> np.ndarray:
        """
        Project a set of points onto a line, represented via one point on the line and the line direction.
        Note that if the instance is a ray, then points behind the ray will get np.nan as projected point.
        :param points_to_project: The input points that are to be projected onto the line; can also be a single point.
                                  Shape (3) or (nx3)
        :return np array of points projected onto the line. Shape same as input, so (3) or (nx3).
        """
        # Projection of the vector onto the line direction
        direction_dot = self.compute_points_distance_on_line(points_to_project)

        # Points that are behind a ray get nan values assigned. This ensures that they can be
        # distinguished from points being projected directly onto the ray origin.
        if self.is_ray:
            direction_dot[direction_dot < 0] = np.nan

        # Distinguish between case where points_to_project is only a single point vs multiple
        if points_to_project.ndim == 2:
            direction_dot = np.expand_dims(direction_dot, axis=-1)
        projections_onto_direction = direction_dot * self.line_direction

        # Add this projection to the original point on the line to get the projection on the line
        projected_points = self.point_on_line + projections_onto_direction

        return projected_points

    def project_vectors_to_line(self, vectors_to_project: np.ndarray) -> np.ndarray:
        """
        Project a set of vectors onto a line defined by its directional vector.
        :param vectors_to_project: The input vectors that are to be projected onto the line; can also be a single vector.
                                   Shape (3) or (nx3)
        :return: np array of vectors projected onto the line. Shape same as input, so (3) or (nx3).
                 The vectors are not normalized.
        """
        # Projection of the vector onto the normal
        line_dot = np.dot(vectors_to_project, self.line_direction)
        # Distinguish between case where vectors_to_project is only a single vector vs multiple
        if vectors_to_project.ndim == 2:
            line_dot = np.expand_dims(line_dot, axis=-1)

        # Project the vectors onto the line direction
        return line_dot * self.line_direction

    def intersect_line(self, other_line: "Line") -> Tuple[np.ndarray, np.ndarray]:
        """
        cf. docs of intersect_lines()
        """
        return self.intersect_lines(self, other_line)

    @staticmethod
    def intersect_lines(line1: "Line", line2: "Line") -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the intersection of two lines in 3D space.
        Two arbitrary lines never actually intersect each other in 3D space,
        and even if the lines are chosen to be in the same plane,
        numerical inaccuracies may still not find an exact
        intersection, so this method computes the two points that lie on the first and second line respectively and
        that are closest to the second and first line respectively. If the points do actually intersect,
        these two points should be (close to) identical.
        :param line1: First line (to be intersected with line2)
        :param line2: Second line (to be intersected with line1)
        :return: Tuple of two 3D numpy points: 1) point on first line that's closest to second line
                                               2) point on second line that's closest to first line
        :raise ValueError: If the two lines are parallel
        """
        p1, p2 = line1.get_point(), line2.get_point()
        d1, d2 = line1.get_direction(), line2.get_direction()

        # Define vector w from p2 to p1
        w0 = p1 - p2

        # Calculate coefficients of the system
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        d = np.dot(d1, w0)
        e = np.dot(d2, w0)

        # Solve the system of equations to find t and s
        denominator = a * c - b * b
        if denominator < 1e-6:
            raise ValueError("The lines are (close to) parallel, so there's no unique closest point.")

        t = (b * e - c * d) / denominator
        s = (a * e - b * d) / denominator

        # Closest point on Line 1
        closest_point_line1 = p1 + t * d1
        # Closest point on Line 2 (for reference, not necessarily needed)
        closest_point_line2 = p2 + s * d2

        return closest_point_line1, closest_point_line2

    def intersect_line_and_line_segments(self, line_segments):
        """
        Intersect a line with a set of line segments. Since it's 3D, the "intersection" will check for the closest point
        on each line segment and check if it's within the segment's limits. There can be from zero up to n intersection
        points, where n would be the number of line segments (although having n intersection points would kinda mean that
        the line segments go zick-zack from one side of the line to the other, or something like that).
        :param line_segments: List of line segments, each segment consisting of two 3D numpy points that
                              define beginning and end point of line segment.
        :return: Tuple: 1) List of 3D numpy intersection points.
                        2) List of line segment indices that correspond to the intersection points.
        """
        # Keep results in list
        intersection_points = []
        intersection_segment_indices = []
        # Loop over each line segment and check individually for intersection with line
        for segment_idx, line_segment in enumerate(line_segments):
            line_segment_point = line_segment[0]
            line_segment_direction = line_segment[1]-line_segment[0]
            # Find point on line segment that's closest to line
            try:
                _, segment_intersection_point = self.intersect_line(Line(line_segment_point, line_segment_direction))
            # If the line segment is (very close) to parallel to the line, we skip it
            except ValueError:
                continue
            intersection_vector = segment_intersection_point - line_segment_point
            # If the intersection point is closer to the first line segment point than the second point is,
            # and if the intersection point is on the positive side of the first segment point, i.e.,
            # on the same side as the second, i.e., the vectors between int point and first point and
            # between second point and first point align, then the intersection points is within the
            # limits of the line segment.
            if np.linalg.norm(intersection_vector) <= np.linalg.norm(line_segment_direction) and np.dot(intersection_vector, line_segment_direction) > 0:
                intersection_points.append(segment_intersection_point)
                intersection_segment_indices.append(segment_idx)
        # Return the 3D intersection points and the corresponding list of line segment indices.
        return intersection_points, intersection_segment_indices
