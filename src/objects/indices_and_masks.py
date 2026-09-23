"""
IndicesAndMasks class to support unified handling of integer index and boolean mask arrays.
Only requires numpy.

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

from __future__ import annotations

from pathlib import Path
from typing import Union, Dict, List, TYPE_CHECKING

import numpy as np
import warnings

if TYPE_CHECKING:
    from mesh import Mesh


class IndicesAndMasks:
    """
    Class with only static method. Not allowed to be instantiated. The main purpose is to handle arrays of
    integer indices and boolean masks. Handling includes converting from index to boolean arrays and vice versa,
    loading/exporting them, joining, intersecting, and subtracting several of them, etc. The primary application
    meant for this class is to index Mesh vertices, so some methods also accept a Mesh as alternative input to
    specify the total number of indexable entities. If you want to index the triangles or edges instead, you will
    need to get that information from the mesh. The use of this class is agnostic to the type of object to be indexed,
    you may just need to provide the total number of entities you want to index, since this information is not
    contained in an index array.
    The main format of the boolean and integer arrays is numpy, but lists are also accepted.
    It does not work with torch.
    Index arrays could look like this [1, 4, 23, 45, ...], whereas
    Boolean masks could look like this [True, False, False, True, True, ...].
    Index arrays only include those indices that would be True in a corresponding Boolean mask,
    so they can be a bit sparser. They don't contain information about how many elements there are in total
    (except that it must be at least the highest index contained in the array + 1). However, index arrays also allow
    an irregular structure with unsorted and duplicate indices, so you could index element 4 multiple times and at
    different places, which the regular structure of boolean masks doesn't allow. This is why the file export and
    import uses indices, not masks, which only requires the user to keep track of the total number of elements
    (any array that the user wants to index contains that information through its length). Boolean masks still have
    the advantage of efficiently checking if an element is indexed or not (should be O(1), at least amortized).
    """
    def __new__(cls, *args, **kwargs):
        raise TypeError(f"{cls.__name__} cannot be instantiated")

    @staticmethod
    def convert_mask_to_indices(mask: Union[np.ndarray, List]) -> np.ndarray:
        """
        Convert a boolean mask array to an array of integer indices. The indices are sorted and unique.
        """
        return np.where(mask)[0]

    @staticmethod
    def get_num_indexed(mask_or_indices: Union[np.ndarray, List], count_only_unique: bool = True) -> int:
        """
        Returns the number of indices in an index or mask array, by default the number of unique indices,
        but you can also choose to count all indices. Unlike get_total_num_indices(), this method is accurate for both,
        index arrays and masks.
        :param mask_or_indices: boolean mask (will count number True entries) or integer array (will count number of
                                (unique) entries)
        :param count_only_unique: Set to False to count all entries in an index array instead of the unique ones
        :return: integer representing the number of indexed values.
        """
        return len(IndicesAndMasks.get_indices(mask_or_indices, make_unique=count_only_unique))

    @staticmethod
    def contains_duplicates(indices: Union[np.ndarray, List]) -> bool:
        return IndicesAndMasks.get_num_indexed(indices, count_only_unique=False) != IndicesAndMasks.get_num_indexed(indices, count_only_unique=True)

    @staticmethod
    def get_total_num_indices(indices_or_mask: Union[np.ndarray, List]) -> int:
        """
        Compute/estimate the total number of elements to be indexed by the given array. For a boolean mask,
        this is easy, it's just the length of the array, but for an index array, we can only estimate it to be the
        maximum index in the array plus 1. Note the difference to get_num_indexed(), which only counts how many
        elements are actually indexed (not how many there are in total).
        :param indices_or_mask: index or mask array
        :return: Integer representing the total number of elements to be indexed.
                 This can be inaccurate for index arrays.
        """
        if IndicesAndMasks.is_mask(indices_or_mask):
            return len(indices_or_mask)
        elif IndicesAndMasks.is_indices(indices_or_mask):
            if len(indices_or_mask) == 0:
                return 0
            else:
                return np.max(indices_or_mask) + 1
        else:
            raise TypeError(f"Required indices or mask array, received {indices_or_mask}")


    @staticmethod
    def _get_total_num_indices(num_indices_or_mesh: Union["Mesh", int] = None,
                               indices_or_mask: Union[np.ndarray, List] = None) -> int:
        """
        Method meant for internal use, since multiple other methods use it. Based on an integer or mesh,
        it returns the total number of entities to be indexed, which can also be decided based on a index or mask
        array instead, or even complemented.
        :param num_indices_or_mesh: Either the integer that already specifies the number we're looking for or a
                                    mesh from which the number of vertices is used. If you want to index the triangles
                                    or edges instead, you will need to get that information from the mesh, and provide
                                    the integer here.
        :param indices_or_mask: Optionally provide index or mask array to estimate the total number of elements.
                                If num_indices_or_mesh is not provided, this must be provided.
        :return: integer representing the total number of indexed values.
        """
        if num_indices_or_mesh is None:
            if indices_or_mask is None:
                raise AssertionError("If no num indices are provided, you must provide the indices or mask array.")
            return IndicesAndMasks.get_total_num_indices(indices_or_mask)
        else:
            if isinstance(num_indices_or_mesh, (int, np.integer)):
                num_indices = num_indices_or_mesh
            elif hasattr(num_indices_or_mesh, "get_num_vertices"):
                num_indices = num_indices_or_mesh.get_num_vertices()
            else:
                raise TypeError(f"Unknown input. Required number of indices or Mesh. Got: {num_indices_or_mesh}.")
            if indices_or_mask is None:
                return num_indices
            else:
                array_num_indices = IndicesAndMasks.get_total_num_indices(indices_or_mask)
                if num_indices < array_num_indices:
                    raise AssertionError("You provided the number of indices as int or mesh, and you provided"
                                         "the actual indices or mask array, but this array has indexes higher values"
                                         "than the num_indices_or_mesh argument provides. Check.")
                return num_indices


    @staticmethod
    def convert_indices_to_mask(indices: Union[np.ndarray, List],
                                num_indices_or_mesh: Union["Mesh", int] = None) -> np.ndarray:
        """
        Convert an array of integer indices to a boolean mask. Note that you can lose some information here, since
        any order or duplicates in the index array are lost.
        :param indices: Numpy array of integer indices.
        :param num_indices_or_mesh: Total number of indices. The index array may not go until the highest element,
                                    so you need to provide that highest index+1 to get the correct mask. E.g.,
                                    a mesh may have 5 vertices, and index array [0, 2, 3] should be converted to
                                    [True, False, True, True, False], but the last element would be missing without you
                                    specifying that it's num_indices=5 vertices in total. You can also just provide the
                                    Mesh, then it is assumed that you want to index the vertices, so the total number
                                    of vertices is used.
        :return: Numpy boolean mask array.
        """
        num_indices = IndicesAndMasks._get_total_num_indices(num_indices_or_mesh=num_indices_or_mesh, indices_or_mask=indices)
        mask = np.zeros(num_indices, dtype=bool)
        if len(indices) > 0:
            mask[indices] = True
        if IndicesAndMasks.get_num_indexed(mask) != IndicesAndMasks.get_num_indexed(indices, count_only_unique=False):
            warnings.warn("The index array you converted to a mask is not unique, so the mask does not contain all "
                          "the information about it. ")
        return mask

    @staticmethod
    def is_indices(indices_or_mask: Union[np.ndarray, List]) -> bool:
        """
        Returns True if the input is an integer index array, so if it's of the format [0, 2, 5, ...].
        Note that we only check that there's only integers. We don't check if the indices make sense,
        so we also allow negative numbers, as well as indices that are larger than the array you want to index.
        We also allow an array of length 0.
        """
        if len(indices_or_mask) == 0:
            return True
        return np.issubdtype(np.asarray(indices_or_mask).dtype, np.integer)

    @staticmethod
    def is_mask(indices_or_mask: Union[np.ndarray, List]) -> bool:
        """
        Returns True if the input is a boolean mask array, so if it's of the format [True, False, True, True, ...].
        Note that we only check that there's only booleans. We don't check if the length of the mask is equivalent to
        the length of the array you want to index. We don't allow an array of length 0 as a mask,
        unlike for index arrays.
        """
        return np.issubdtype(np.asarray(indices_or_mask).dtype, bool)

    @staticmethod
    def get_mask(indices_mask_or_path: Union[Union[Path, str], Union[np.ndarray, List]],
                 num_indices_or_mesh: Union[int, "Mesh"] = None) -> np.ndarray:
        """
        Get a boolean mask array from an input index or mask array, or a path to an index array.
        :param indices_mask_or_path: Three allowed cases:
                                    1) boolean mask array -> no conversion
                                    2) index int array -> converted to boolean mask array (potentially lossy
                                       conversion, cf. convert_indices_to_mask() method).
                                    3) Path to index array -> file loaded and index array converted like in case 2)
        :param num_indices_or_mesh: Total number of indices or mesh, cf. docs of convert_indices_to_mask.
        :return Numpy boolean mask array.
        """
        # A path is imported and converted to a mask (since the load() function only returns index arrays)
        if isinstance(indices_mask_or_path, (str, Path)):
            indices = IndicesAndMasks.load(indices_mask_or_path)
            return IndicesAndMasks.convert_indices_to_mask(indices, num_indices_or_mesh=num_indices_or_mesh)
        # For an array, we first check if it is a index array or a mask
        else:
            indices_or_mask = np.asarray(indices_mask_or_path)
            # An index array is converted, using the additional information how many elements there are in total
            if IndicesAndMasks.is_indices(indices_or_mask):
                return IndicesAndMasks.convert_indices_to_mask(indices_or_mask, num_indices_or_mesh=num_indices_or_mesh)
            # A mask is not directly returned. We possibly extend it if to the num_indices provided before returning it.
            elif IndicesAndMasks.is_mask(indices_or_mask):
                num_indices = IndicesAndMasks._get_total_num_indices(
                    num_indices_or_mesh=num_indices_or_mesh, indices_or_mask=indices_or_mask)
                if num_indices > IndicesAndMasks.get_total_num_indices(indices_or_mask):
                    mask_extended = np.zeros(num_indices)
                    mask_extended[indices_or_mask] = True
                    return mask_extended
                else:
                    return indices_or_mask
            else:
                raise TypeError(f"{indices_mask_or_path} is not a valid mask or index type")

    @staticmethod
    def get_indices(indices_mask_or_path: Union[Union[str, Path], Union[np.ndarray, List]],
                    make_unique: bool = False) -> np.ndarray:
        """
        Get a integer index array from an input index or mask array, or a path to an index array.
        :param indices_mask_or_path: Three allowed cases:
                                    1) boolean mask array -> converted to index array
                                       (cf. convert_mask_to_indices() method).
                                    2) index int array -> no conversion
                                    3) Path to index array -> file loaded and returned without further conversion.
        :param make_unique: If True, the array is sorted and made unique (using np.unique)
        :return Numpy integer index array (not necessarily sorted and unique, but can be forced to be via make_unique).
        """
        # A path is imported and doesn't require conversion (since the load() function already returns index arrays)
        if isinstance(indices_mask_or_path, (str, Path)):
            indices = IndicesAndMasks.load(indices_mask_or_path)
        # An array is kept if it's already and index array or converted if it's a mask
        else:
            indices_or_mask = np.asarray(indices_mask_or_path)
            if IndicesAndMasks.is_indices(indices_or_mask):
                indices = indices_or_mask
            elif IndicesAndMasks.is_mask(indices_or_mask):
                # this is always unique, so we can directly return it
                return IndicesAndMasks.convert_mask_to_indices(indices_or_mask)
            else:
                raise TypeError(f"{indices_mask_or_path} is not a valid mask or index type")
        # Optionally make the index array unique
        if make_unique:
            indices = np.unique(indices)
        return indices

    @staticmethod
    def get_mask_or_indices(
            indices_mask_or_path: Union[Union[str, Path], Union[np.ndarray, List]],
            return_as_indices: Union[bool, None] = None, num_indices_or_mesh: Union[int, "Mesh"] = None,
            make_unique: bool = False) -> np.ndarray:
        """
        This method supports loading and converting index or mask arrays into one another. It's probably the most
        flexible method.
        :param indices_mask_or_path: index or mask array or path to it.
        :param return_as_indices: Set to True to get an index array returned, False to get a mask array.
                                  Default is the input type, except when it's a path, then index array is returned.
        :param num_indices_or_mesh: Total number of indices or Mesh, cf. docs of convert_indices_to_mask.
        :param make_unique: Set to True to make the index array unique (if index array is returned).
        :return: Index or boolean array.
        """
        if return_as_indices is None:
            if isinstance(indices_mask_or_path, (str, Path)):
                return_as_indices = True
            else:
                return_as_indices = IndicesAndMasks.is_indices(indices_mask_or_path)
        if return_as_indices:
            return IndicesAndMasks.get_indices(indices_mask_or_path, make_unique=make_unique)
        else:
            return IndicesAndMasks.get_mask(indices_mask_or_path, num_indices_or_mesh=num_indices_or_mesh)


    @staticmethod
    def invert_mask(mask: Union[np.ndarray, List]) -> np.ndarray:
        """
        Simple mask inversion (np logical not)
        """
        return np.logical_not(mask)

    @staticmethod
    def invert_indices(indices: Union[np.ndarray, List], num_indices_or_mesh: Union[int, "Mesh", None] = None) -> np.ndarray:
        """
        Invert an array of integer indices. The inversion does not conserve any kind of order or duplicates.
        It only returns all indices in order that are not indexed in the given array.
        :param indices: Numpy array of integer indices.
        :param num_indices_or_mesh: Total number of indices or Mesh, cf. docs of convert_indices_to_mask.
        :return: Numpy array of integer indices not indexed by the input array.
        """
        mask = IndicesAndMasks.convert_indices_to_mask(indices, num_indices_or_mesh=num_indices_or_mesh)
        mask_inverted = IndicesAndMasks.invert_mask(mask)
        return IndicesAndMasks.convert_mask_to_indices(mask_inverted)

    @staticmethod
    def invert(indices_or_mask: Union[np.ndarray, List], num_indices_or_mesh: Union[int, "Mesh", None] = None) -> np.ndarray:
        """
        Invert indices or mask, cf. docs for invert_mask() and invert_indices().
        """
        if IndicesAndMasks.is_indices(indices_or_mask):
            return IndicesAndMasks.invert_indices(indices_or_mask, num_indices_or_mesh=num_indices_or_mesh)
        elif IndicesAndMasks.is_mask(indices_or_mask):
            return IndicesAndMasks.invert_mask(indices_or_mask)
        else:
            raise TypeError(f"{indices_or_mask} is not a valid mask or index type.")

    @staticmethod
    def get_num_indices_from_multiple_arrays(
            *index_arrays_or_masks: Union[np.ndarray, List], num_indices_or_mesh: int = None) -> int:
        """
        Similar to get_num_indices(), but supports multiple index/mask arrays, so if num_indices is not known,
        the max is taken over all these arrays.
        """
        array_num_indices = np.max([IndicesAndMasks.get_total_num_indices(index_array)
                                    for index_array in index_arrays_or_masks])
        if num_indices_or_mesh is None:
            num_indices = array_num_indices
        else:
            num_indices = IndicesAndMasks._get_total_num_indices(num_indices_or_mesh=num_indices_or_mesh)
            if num_indices < array_num_indices:
                raise AssertionError("Provided num_indices is smaller than required by the given arrays.")
        return num_indices

    @staticmethod
    def join(*index_arrays_or_masks: Union[np.ndarray, List],
             num_indices_or_mesh: Union[int, "Mesh", None] = None, return_as_indices: Union[bool, None] = None,
             make_unique: bool = True) -> np.ndarray:
        """
        Join multiple index or mask arrays into one (OR/UNION)
        @param index_arrays_or_masks: Zero or more index arrays of either bool or int type.
                                      (Might also be possible with paths, but no guarantee).
        @param num_indices_or_mesh: Total number of indices or Mesh, cf. docs of convert_indices_to_mask.
        @param return_as_indices: Set to True to get indices as return and to False to get a mask as return.
                                  If not specified, the return type is the same as the first input array.
        @param make_unique: Set to False to have the joined index arrays not unique.
        @return: joined integer index or mask boolean array.
        """
        index_arrays_or_masks = [IndicesAndMasks.get_mask_or_indices(indices_or_mask)
                                 for indices_or_mask in index_arrays_or_masks]
        # If no arrays are given, we return an empty index array or mask
        if len(index_arrays_or_masks) == 0:
            return IndicesAndMasks.get_mask_or_indices(
                [], return_as_indices=return_as_indices, num_indices_or_mesh=num_indices_or_mesh)
        if return_as_indices is None:
            return_as_indices = IndicesAndMasks.is_indices(index_arrays_or_masks[0])

        # If an index array should be returned, we simply convert all input arrays to index arrays and concatenate them.
        # The concatenated array is optionally made unique.
        if return_as_indices:
            joined_indices = np.concatenate([IndicesAndMasks.get_indices(
                index_array, make_unique=False) for index_array in index_arrays_or_masks]).astype(int)
            return IndicesAndMasks.get_indices(joined_indices, make_unique=make_unique)
        # If a mask should be returned, we convert all input arrays to masks, using the largest num_indices we find.
        # The masks are then combined via an OR operation.
        else:
            num_indices = IndicesAndMasks.get_num_indices_from_multiple_arrays(
                *index_arrays_or_masks, num_indices_or_mesh=num_indices_or_mesh)

            return np.logical_or.reduce([
                IndicesAndMasks.get_mask(indices_or_mask, num_indices_or_mesh=num_indices)
                for indices_or_mask in index_arrays_or_masks])

    @staticmethod
    def intersect(*index_arrays_or_masks: Union[np.ndarray, List],
                  num_indices_or_mesh: Union[int, "Mesh", None] = None,
                  return_as_indices: Union[bool, None] = None) -> np.ndarray:
        """
        Find intersection (AND) between one or more index arrays, so all indices provided by all given index arrays.
        The returned array is always made unique, since the intersection uses conversion to boolean arrays.
        @param index_arrays_or_masks: Multiple index arrays to be intersected. Empty index array is return if no index array is
                             given.
        @param num_indices_or_mesh: Total number of indices or Mesh, cf. docs of convert_indices_to_mask.
        @param return_as_indices: Set to True to get indices as return and to False to get a mask as return.
                                  If not specified, the return type is the same as the first input array.
        @return: intersected integer index or boolean mask array.
        """
        if len(index_arrays_or_masks) == 0:
            return IndicesAndMasks.get_mask_or_indices(
                [], num_indices_or_mesh=num_indices_or_mesh, return_as_indices=return_as_indices,
                make_unique=True)
        if return_as_indices is None:
            return_as_indices = IndicesAndMasks.is_indices(index_arrays_or_masks[0])
        if len(index_arrays_or_masks) == 1:
            return IndicesAndMasks.get_mask_or_indices(
                index_arrays_or_masks[0], num_indices_or_mesh=num_indices_or_mesh, return_as_indices=return_as_indices,
                make_unique=True)

        num_indices = IndicesAndMasks.get_num_indices_from_multiple_arrays(
            *index_arrays_or_masks, num_indices_or_mesh=num_indices_or_mesh)

        # To intersection first converts all input arrays to masks, and then combines them via an AND operation.
        combined_mask = np.logical_and.reduce([
            IndicesAndMasks.get_mask(indices_or_mask, num_indices_or_mesh=num_indices)
            for indices_or_mask in index_arrays_or_masks])
        return IndicesAndMasks.get_mask_or_indices(combined_mask, return_as_indices=return_as_indices)

    @staticmethod
    def subtract(first_indices_or_mask: Union[np.ndarray, List],
                 *subtract_index_arrays_or_masks: Union[np.ndarray, List],
                 num_indices_or_mesh: Union[int, "Mesh"], return_as_indices: Union[bool, None] = None,
                 make_unique: bool = False) -> np.ndarray:
        """
        Subtract multiple index arrays from one.
        @param first_indices_or_mask: Main index or bool array from which other arrays are subtracted
        @param subtract_index_arrays_or_masks: Zero or more index or bool arrays that are subtracted from
                                               first_indices_or_mask.
        @param num_indices_or_mesh: Total number of indices or Mesh, cf. docs of convert_indices_to_mask.
        @param return_as_indices: Set to True to get indices as return and to False to get a mask as return.
                                  If not specified, the return type is the same as the first input array.
        @return: Subtracted index or bool array.
        """
        if len(subtract_index_arrays_or_masks) == 0:
            return IndicesAndMasks.get_mask_or_indices(
                first_indices_or_mask, return_as_indices=return_as_indices, num_indices_or_mesh=num_indices_or_mesh,
                make_unique=make_unique)
        if return_as_indices is None:
            return_as_indices = IndicesAndMasks.is_indices(first_indices_or_mask)

        num_indices = IndicesAndMasks.get_num_indices_from_multiple_arrays(
            first_indices_or_mask, *subtract_index_arrays_or_masks, num_indices_or_mesh=num_indices_or_mesh)

        # We combine all the arrays to be subtracted from the first array via a join (OR, UNION).
        subtract_joined_mask = IndicesAndMasks.join(
            *subtract_index_arrays_or_masks, return_as_indices=False, num_indices_or_mesh=num_indices)
        # If an index array should be returned, we make sure the first array is one, then loop over its indices and
        # only keep those not indexed by the combined subtraction mask.
        if return_as_indices:
            first_indices = IndicesAndMasks.get_indices(first_indices_or_mask, make_unique=make_unique)
            return np.asarray([idx for idx in first_indices if not subtract_joined_mask[idx]])
        # If a mask should be returned, we make sure the first array is one, then we intersect it with the
        # inverted combined subtraction mask, so that only those values stay True not indexed by the
        # joined subtraction mask.
        else:
            first_mask = IndicesAndMasks.get_mask(first_indices_or_mask, num_indices_or_mesh=num_indices)
            return IndicesAndMasks.intersect(first_mask, IndicesAndMasks.invert_mask(subtract_joined_mask))


    @staticmethod
    def load(path_to_indices: Union[str, Path]) -> np.ndarray:
        """
        Load an array of integer indices. This method also supports loading boolean masks, but they're always
        converted to an index array. You probably don't have to use this method.
        You can just use get_mask() or get_indices().
        :param path_to_indices: Path to an index array (string or pathlib Path), must be a text file containing the
                                indices or boolean entries separated by commas and/or new lines.
        """

        def to_int_or_bool(my_str: str) -> Union[int, bool]:
            try:
                return int(my_str)
            except ValueError:
                if my_str.lower() in ["t", "true", "y", "yes", "yep", "aha"]:
                    return True
                return False

        indices_text = Path(path_to_indices).read_text()
        if len(indices_text) == 0:
            indices = []
        else:
            chars_to_remove = "[] \"\'"
            for char_to_remove in chars_to_remove:
                indices_text = indices_text.replace(char_to_remove, "")
            indices_text = indices_text.replace("\n", ",")
            indices_text = indices_text.replace(",,", ",")
            indices_split = indices_text.split(",")
            indices = [to_int_or_bool(idx) for idx in indices_split]
        return IndicesAndMasks.get_indices(indices)

    @staticmethod
    def export(indices_or_mask, path_to_indices: Union[str, Path], overwrite: bool = True) -> None:
        """
        Export an index or mask array to a text file. Masks are always converted to an index array.
        The saved file is of the format "idx0,idx1,idx2,...,idxn", for example "1,3,2,7,8". Sorting and duplicates
        of index arrays are preserved. Index arrays are the preferred format for storing because of this possibility
        for unsorted and duplicate entries.
        :param indices_or_mask: Numpy array of integer indices or boolean mask
        :param path_to_indices: Path to the file the indices should be saved to.
        :param overwrite: If True, overwrite existing file (default).
        :return None
        """
        path_to_indices = Path(path_to_indices)
        if path_to_indices.exists() and not overwrite:
            warnings.warn("File already exists. Use overwrite=True to overwrite it.")
            return
        indices = IndicesAndMasks.get_indices(indices_or_mask, make_unique=False)
        path_to_indices.write_text(",".join([str(idx) for idx in indices]))

    @staticmethod
    def get_new_indices_after_index_removal(
            original_indices_or_mask : np.ndarray, removed_indices_or_mask: np.ndarray) -> np.ndarray:
        """
        This method computes the indices that still remain after removing some indices.
        E.g., if you have some mesh vertices indexed, but then you remove some vertices from the mesh,
        this method gives you the same vertex selection on the new mesh.
        This method preserves order of provided indices. However, note that if you input the original as indices,
        you may get fewer numbers as output, and the values may be reduced. E.g., if the original indices are
        [0, 2, 3, 8, 9], and you removed [1, 5, 9], you will get [0, 1, 2, 6] as output.
        @param original_indices_or_mask: The indices or mask of interest before the nay indices were removed.
        @param removed_indices_or_mask: The indices or mask of all deleted vertices.
        @return: Index or mask (same type as original_indices_or_mask) numpy array that remain after the removal.

        """
        def get_bool_len(idx_array):
            return len(idx_array) if IndicesAndMasks.is_mask(idx_array) else np.max(idx_array).astype(int) + 1

        # To convert from int to bool arrays, we set the length of the indexed values to be one bigger than the max element
        # in any of the two arrays (if they're already bool arrays, we just take the max length). This allows to convert
        # index arrays without having to specify what the total number of indices (e.g. mesh vertices) actually is;
        # it doesn't matter in this context.
        num_indices = int(max(get_bool_len(original_indices_or_mask), get_bool_len(removed_indices_or_mask)))

        # To preserve order and possible duplicates in the original indices, we loop through them and reduce them by the
        # number of deleted indices that come before.
        # This method is still a bit more inefficient than using bool arrays,
        # so we only do this if the user inputs and int array.
        if IndicesAndMasks.is_indices(original_indices_or_mask):
            removed_mask = IndicesAndMasks.get_mask(removed_indices_or_mask)
            new_indices = np.asarray([original_ind - np.sum(removed_mask[:original_ind]) for original_ind in
                                      original_indices_or_mask if not removed_mask[original_ind]])
            return new_indices

        # If the user inputs a mask, the order of the original indices needs not be preserved,
        # hence we simply index the mask via the indices that weren't removed.

        remaining_indices = IndicesAndMasks.invert(removed_indices_or_mask, num_indices)

        # new mask can be found by indexing the original bool array via the remaining indices
        return original_indices_or_mask[remaining_indices]

    @staticmethod
    def convert_inner_frame_mask_to_outer_frame(
            inner_frame_mask: np.ndarray, outer_frame_mask: np.ndarray) -> np.ndarray:
        """
        Given a mask in some inner frame, you extrapolate its entries to an outer frame.
        E.g., suppose you have the inner frame mask [True, False, True] and a supporting
        outer frame mask [False, True, False, True, True]. The True values in the outer
        frame mask specify where the entries of the inner frame mask should be extrapolated
        to. The result would then be [False, True(1), False, False(2), True(3)], where (i)
        after the entry specifies which entry of the inner frame mask this corresponds to.
        The False entries of the outer frame mask stay False.
        :param inner_frame_mask: Numpy boolean mask array. To be extrapolated.
        :param outer_frame_mask: Numpy boolean mask array. Defines extrapolation of inner_frame_mask.
                                 Number of True entries must equal length of inner_frame_mask.
        :return: Converted numpy boolean mask.
        """
        combined_mask = np.zeros_like(outer_frame_mask)
        combined_inner_frame = combined_mask[outer_frame_mask]
        combined_inner_frame[inner_frame_mask] = True
        combined_mask[outer_frame_mask] = combined_inner_frame
        return combined_mask

    @staticmethod
    def get_old_indices_before_deletion(
            new_indices_or_mask: np.ndarray, deleted_indices_or_mask: np.ndarray) -> np.ndarray:
        """
        This method computes indices before some were deleted, e.g., vertex indices corresponding to a current
        vertex selection, but before some other vertices were deleted. Index order and duplicates are preserved, e.g.
        if the new indices are [1, 1, 0, 2] that resulted from deleting indices [1, 2], then the old indices
        are [3, 3, 0, 4]. Note that this method cannot recover old indices that were among the deleted ones.
        :param new_indices_or_mask: index or mask array of the current index selection.
        :param deleted_indices_or_mask: index or mask array of the indices that were deleted.
        :return: index or mask array (only mask if new_indices_or_mask and deleted_indices_or_mask are both masks)
                 of the selection before the deletion of deleted_indices_or_mask, so when calling
                 get_new_indices_after_deletion(method_return, deleted_indices_or_mask),
                 you should get new_indices_or_mask out.
        """
        if IndicesAndMasks.get_num_indexed(deleted_indices_or_mask) == 0:
            return np.array(new_indices_or_mask)
        if IndicesAndMasks.is_indices(deleted_indices_or_mask):
            new_index_max = IndicesAndMasks.get_total_num_indices(new_indices_or_mask)
            max_len = int(new_index_max + np.max(deleted_indices_or_mask) + 1)
            deleted_mask = IndicesAndMasks.get_mask(deleted_indices_or_mask, num_indices_or_mesh=max_len)
        else:
            max_len = len(deleted_indices_or_mask)
            deleted_mask = deleted_indices_or_mask

        remaining_indices = IndicesAndMasks.get_indices(IndicesAndMasks.invert_mask(deleted_mask))
        old_indices = remaining_indices[new_indices_or_mask]
        if IndicesAndMasks.is_mask(new_indices_or_mask) and IndicesAndMasks.is_mask(deleted_indices_or_mask):
            return IndicesAndMasks.get_mask(old_indices, num_indices_or_mesh=max_len)
        return old_indices

    @staticmethod
    def is_unique(index_array: Union[list, np.ndarray]) -> bool:
        if IndicesAndMasks.is_mask(index_array):
            warnings.warn("Masks are always unique")
            return True
        return len(IndicesAndMasks.get_indices(index_array, make_unique=True)) == len(index_array)

