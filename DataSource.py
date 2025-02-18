import json
import os
from tqdm import tqdm

import numpy as np
from PIL import Image

import grain.python as grain


def combine_data(annotation_files: list[str], ground_truth_file: str) -> list[dict]:
    """
    """
    # region DEFAULT VALUE OF MISSING ANNOTATION
    _missing_value: int = -1
    _is_one_hot: bool = False
    _num_class: int
    _missing_label: list[int] | int
    # endregion

    all_filenames = []
    annotators = []

    # load json data
    for annotation_file in tqdm(
        iterable=annotation_files,
        desc='Load annotations',
        ncols=80,
        colour='green',
        leave=False
    ):
        with open(file=annotation_file, mode='r') as f:
            # load a list of dictionaries
            annotations = json.load(fp=f)

        # initialise a dict: key=file, value=label for each annotator
        annotator = {}

        for annotation in annotations:
            file_path = annotation['file']

            if file_path not in all_filenames:
                all_filenames.append(file_path)

            annotator[file_path] = annotation['label']

            # check if one_hot or not
            if isinstance(annotation['label'], int):
                continue
            elif isinstance(annotation['label'], list):
                _is_one_hot = True
                _num_class = len(annotation['label'])
            else:
                raise ValueError(
                    'Unknown label format. Expect either integer or list of floats, '
                    f'but found {annotation["label"].__class__}'
                )

        annotators.append(annotator)

    # set missing labels
    if _is_one_hot:
        _missing_label = [_missing_value] * _num_class
    else:
        _missing_label = _missing_value

    with open(file=ground_truth_file, mode='r') as f:
        ground_truth_annotations = json.load(fp=f)

    ground_truth_annotations = {
        gt_annotation['file']: gt_annotation['label']
            for gt_annotation in ground_truth_annotations
    }

    # initialise a list to store all the json data from all the provided datasets
    consolidation = []

    for filename in tqdm(iterable=ground_truth_annotations, desc='make dataset', ncols=80, leave=False):
        # file_path = os.path.join(root, filename) if root is not None else filename
        # record = dict(file=file_path.encode('ascii'), label=[], ground_truth=_missing_label)
        record = dict(file=filename, label=[], ground_truth=_missing_label)

        for annotator in annotators:
            record['label'].append(annotator.get(filename, _missing_label))

        # set ground truth
        record['ground_truth'] = ground_truth_annotations.get(filename)

        consolidation.append(record)
    
    return consolidation


class ImageDataSource(grain.RandomAccessDataSource):
    def __init__(
        self,
        annotation_files: list[str],
        ground_truth_file: str,
        root: str = None
    ) -> None:
        """make the dataset from multiple annotation files.

        Each file may contain only a subset of the whole dataset.
        If one annotator does not label a sample, the label will be set to -1.

        Args:
            annotation_files: list of pathes to the json files of annotators
            ground_truth_file: path to the json file of ground truth
            root: the directory to the dataset folder
            shape: the new shape (width and height) of the resized image

        Returns:
            dataset:
        """
        self.root = root if root is not None else ''
        self._data = combine_data(
            annotation_files=annotation_files,
            ground_truth_file=ground_truth_file
        )

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """
        """
        # load images
        x = Image.open(fp=os.path.join(self.root, self._data[idx]['file']))
        x = np.array(object=x)

        y = np.array(object=self._data[idx]['ground_truth'], dtype=np.int32)
        t = np.array(object=self._data[idx]['label'], dtype=np.int32)

        return dict(
            filename=self._data[idx]['file'],
            image=x,
            label=t,
            ground_truth=y
        )

    def __len__(self) -> int:
        return len(self._data)