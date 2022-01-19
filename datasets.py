import itertools
from copy import deepcopy
import numpy as np
from torch.utils.data.dataset import Dataset
from utils import get_bb


''' Utility function for patch creation '''


def centers_to_slice(voxels, patch_half):
    """
    Function to convert a list of indices defining the center of a patch, to
    a real patch defined using slice objects for each dimension.
    :param voxels: List of indices to the center of the slice.
    :param patch_half: List of integer halves (//) of the patch_size.
    """
    slices = [
        tuple(
            [
                slice(idx - p_len, idx + p_len) for idx, p_len in zip(
                    voxel, patch_half
                )
            ]
        ) for voxel in voxels
    ]
    return slices


def get_slices(masks, patch_size, overlap):
    """
    Function to get all the patches with a given patch size and overlap between
    consecutive patches from a given list of masks. We will only take patches
    inside the bounding box of the mask. We could probably just pass the shape
    because the masks should already be the bounding box.
    :param masks: List of masks.
    :param patch_size: Size of the patches.
    :param overlap: Overlap on each dimension between consecutive patches.

    """
    # Init
    # We will compute some intermediate stuff for later.
    patch_half = [p_length // 2 for p_length in patch_size]
    steps = [max(p_length - o, 1) for p_length, o in zip(patch_size, overlap)]

    # We will need to define the min and max pixel indices. We define the
    # centers for each patch, so the min and max should be defined by the
    # patch halves.
    min_bb = [patch_half] * len(masks)
    max_bb = [
        [
            max_i - p_len for max_i, p_len in zip(mask.shape, patch_half)
        ] for mask in masks
    ]

    # This is just a "pythonic" but complex way of defining all possible
    # indices given a min, max and step values for each dimension.
    dim_ranges = [
        map(
            lambda t: np.concatenate([np.arange(*t), [t[1]]]),
            zip(min_bb_i, max_bb_i, steps)
        ) for min_bb_i, max_bb_i in zip(min_bb, max_bb)
    ]

    # And this is another "pythonic" but not so intuitive way of computing
    # all possible triplets of center voxel indices given the previous
    # indices. I also added the slice computation (which makes the last step
    # of defining the patches).
    patch_slices = [
        centers_to_slice(
            itertools.product(*dim_range), patch_half
        ) for dim_range in dim_ranges
    ]

    return patch_slices


''' Datasets '''


class ImagePatchesDataset(Dataset):
    def __init__(
            self, subjects, labels, rois, patch_size=32,
            overlap=0, balanced=True,
    ):
        # Init
        if type(patch_size) is not tuple:
            self.patch_size = (patch_size,) * 3
        else:
            self.patch_size = patch_size
        if type(overlap) is not tuple:
            self.overlap = (overlap,) * 3
        else:
            self.overlap = overlap
        self.balanced = balanced

        self.subjects = subjects
        self.rois = rois
        self.labels = labels

        # We get the preliminary patch slices (inside the bounding box)...
        slices = get_slices(self.rois, self.patch_size, self.overlap)

        # ... however, being inside the bounding box doesn't guarantee that the
        # patch itself will contain any lesion voxels. Since, the lesion class
        # is extremely underrepresented, we will filter this preliminary slices
        # to guarantee that we only keep the ones that contain at least one
        # lesion voxel.
        patch_slices = [
            (s, i) for i, (label, s_i) in enumerate(
                zip(labels, slices)
            )
            for s in s_i if np.sum(label[s]) > 0
        ]
        bck_slices = [
            (s, i) for i, (label, s_i) in enumerate(
                zip(labels, slices)
            )
            for s in s_i if np.sum(label[s]) == 0
        ]
        n_positives = len(patch_slices)
        n_negatives = len(bck_slices)

        positive_imbalance = n_positives > n_negatives

        if positive_imbalance:
            self.majority = patch_slices
            self.majority_label = np.array([1], dtype=np.uint8)

            self.minority = bck_slices
            self.minority_label = np.array([0], dtype=np.uint8)
        else:
            self.majority = bck_slices
            self.majority_label = np.array([0], dtype=np.uint8)

            self.minority = patch_slices
            self.minority_label = np.array([1], dtype=np.uint8)

        if self.balanced:
            self.current_majority = deepcopy(self.majority)
            self.current_minority = deepcopy(self.minority)

    def __getitem__(self, index):
        if self.balanced:
            if index < (2 * len(self.minority)):
                flip = index >= len(self.minority)
                index = np.random.randint(len(self.current_minority))
                slice_i, case_idx = self.current_minority.pop(index)
                if len(self.current_minority) == 0:
                    self.current_minority = deepcopy(self.minority)
                target_data = self.minority_label
            else:
                flip = (index - 2 * len(self.minority)) >= len(self.majority)
                index = np.random.randint(len(self.current_majority))
                slice_i, case_idx = self.current_majority.pop(index)
                if len(self.current_majority) == 0:
                    self.current_majority = deepcopy(self.majority)
                target_data = self.majority_label
        else:
            flip = False
            if index < len(self.minority):
                slice_i, case_idx = self.minority[index]
                target_data = self.minority_label
            else:
                index -= len(self.minority)
                slice_i, case_idx = self.majority[index]
                target_data = self.majority_label

        data = self.subjects[case_idx]
        none_slice = (slice(None, None),)
        # Patch "extraction".
        if isinstance(data, tuple):
            data = tuple(
                data_i[none_slice + slice_i].astype(np.float32)
                for data_i in data
            )
        else:
            data = data[none_slice + slice_i].astype(np.float32)
        if flip:
            if isinstance(data, tuple):
                data = tuple(
                    np.fliplr(data_i).copy() for data_i in data
                )
            else:
                data = np.fliplr(data).copy()

        return data, target_data

    def __len__(self):
        if self.balanced:
            return len(self.minority) * 4
        else:
            return len(self.minority) + len(self.majority)


class ImageCroppingDataset(Dataset):
    def __init__(
            self, subjects, labels, rois, patch_size=32,
            overlap=0, filtered=True, balanced=True,
    ):
        # Init
        if type(patch_size) is not tuple:
            self.patch_size = (patch_size,) * 3
        else:
            self.patch_size = patch_size
        if type(overlap) is not tuple:
            self.overlap = (overlap,) * 3
        else:
            self.overlap = overlap
        self.filtered = filtered
        self.balanced = balanced

        self.subjects = subjects
        self.rois = rois
        self.labels = labels

        # We get the preliminary patch slices (inside the bounding box)...
        slices = get_slices(self.rois, self.patch_size, self.overlap)

        # ... however, being inside the bounding box doesn't guarantee that the
        # patch itself will contain any lesion voxels. Since, the lesion class
        # is extremely underrepresented, we will filter this preliminary slices
        # to guarantee that we only keep the ones that contain at least one
        # lesion voxel.
        if self.filtered:
            if self.balanced:
                self.patch_slices = [
                    (s, i) for i, (label, s_i) in enumerate(
                        zip(labels, slices)
                    )
                    for s in s_i if np.sum(label[s]) > 0
                ]
                self.bck_slices = [
                    (s, i) for i, (label, s_i) in enumerate(
                        zip(labels, slices)
                    )
                    for s in s_i if np.sum(label[s]) == 0
                ]
                self.current_bck = deepcopy(self.bck_slices)
            else:
                self.patch_slices = [
                    (s, i) for i, (label, s_i) in enumerate(
                        zip(labels, slices)
                    )
                    for s in s_i if np.sum(label[s]) > 0
                ]
        else:
            self.patch_slices = [
                (s, i) for i, s_i in enumerate(slices) for s in s_i
            ]

    def __getitem__(self, index):
        if index < (2 * len(self.patch_slices)):
            flip = index >= len(self.patch_slices)
            if flip:
                index -= len(self.patch_slices)
            slice_i, case_idx = self.patch_slices[index]
            positive = True
        else:
            flip = np.random.random() > 0.5
            index = np.random.randint(len(self.current_bck))
            slice_i, case_idx = self.current_bck.pop(index)
            if len(self.current_bck) == 0:
                self.current_bck = deepcopy(self.bck_slices)
            positive = False

        data = self.subjects[case_idx]
        if self.labels is None:
            labels = positive
        else:
            labels = self.labels[case_idx]
        none_slice = (slice(None, None),)
        # Patch "extraction".
        if isinstance(data, tuple):
            data = tuple(
                data_i[none_slice + slice_i].astype(np.float32)
                for data_i in data
            )
        else:
            data = data[none_slice + slice_i].astype(np.float32)
        target_data = np.expand_dims(labels[slice_i].astype(np.uint8), axis=0)
        if flip:
            if isinstance(data, tuple):
                data = tuple(
                    np.fliplr(data_i).copy() for data_i in data
                )
            else:
                data = np.fliplr(data).copy()
            target_data = np.fliplr(target_data).copy()

        return data, target_data

    def __len__(self):
        if self.filtered and self.balanced:
            return len(self.patch_slices) * 4
        else:
            return len(self.patch_slices)


class ImageDataset(Dataset):
    def __init__(
        self, subjects, labels, rois
    ):
        # Init
        self.subjects = subjects
        self.rois = rois
        self.labels = labels

    def __getitem__(self, index):
        flip = index >= len(self.labels)
        if flip:
            index -= len(self.labels)

        data = self.subjects[index]
        labels = self.labels[index]
        none_slice = (slice(None, None),)
        bb = get_bb(self.rois[index], 1)
        # Patch "extraction".
        if isinstance(data, tuple):
            data = tuple(
                data_i[none_slice + bb].astype(np.float32)
                for data_i in data
            )
        else:
            data = data[none_slice + bb].astype(np.float32)
        target_data = np.expand_dims(labels[bb].astype(np.uint8), axis=0)
        if flip:
            if isinstance(data, tuple):
                data = tuple(
                    np.fliplr(data_i).copy() for data_i in data
                )
            else:
                data = np.fliplr(data).copy()
            target_data = np.fliplr(target_data).copy()

        return data, target_data

    def __len__(self):
        return len(self.labels) * 2


class ImageClassDataset(Dataset):
    """
    This is a training dataset and we only want patches that
    actually have lesions since there are lots of non-lesion voxels
    anyways.
    """
    def __init__(self, cases, labels, rois):
        # Init
        self.labels = labels
        self.cases = cases
        self.rois = rois

    def __getitem__(self, index):
        flip = index >= len(self.labels)
        if flip:
            index -= len(self.labels)
        data = np.expand_dims(self.cases[index].astype(np.float32), axis=0)
        target = np.array([self.labels[index]], dtype=np.uint8)

        # bb = get_bb(self.masks[index])
        # data = self.cases[index][(slice(None),) + bb].astype(np.float32)

        if flip:
            if isinstance(data, tuple):
                data = tuple(
                    np.fliplr(data_i).copy() for data_i in data
                )
            else:
                data = np.fliplr(data).copy()

        return data, target

    def __len__(self):
        return len(self.cases) * 2
