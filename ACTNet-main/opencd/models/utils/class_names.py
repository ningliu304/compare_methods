# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.utils import is_str


def levir_cd_classes():
    """Cityscapes class names for external use."""
    return [
        'unchanged', 'changed'
    ]


def levir_cd_palette():
    """Cityscapes palette for external use."""
    return [[0, 0, 0], [255, 255, 255]]


def s2looking_classes():
    """Cityscapes class names for external use."""
    return [
        'unchanged', 'changed'
    ]


def s2looking_palette():
    """Cityscapes palette for external use."""
    return [[0, 0, 0], [255, 255, 255]]


def whu_cd_classes():
    """Cityscapes class names for external use."""
    return [
        'unchanged', 'changed'
    ]


def whu_cd_palette():
    """Cityscapes palette for external use."""
    return [[0, 0, 0], [255, 255, 255]]

dataset_aliases = {
    'levir_cd': ['levir_cd'],
    's2looking': ['s2looking'],
    'whu_cd': ['whu_cd'],
}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels


def get_palette(dataset):
    """Get class palette (RGB) of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_palette()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels
