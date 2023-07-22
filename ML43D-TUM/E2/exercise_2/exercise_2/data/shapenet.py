"""PyTorch datasets for loading ShapeNet voxels and ShapeNet point clouds from disk"""
import torch
from pathlib import Path
import json
import numpy as np
import trimesh

from exercise_2.data.binvox_rw import read_as_3d_array


class ShapeNetVox(torch.utils.data.Dataset):
    """
    Dataset for loading ShapeNet Voxels from disk
    """

    num_classes = 13  # we'll be performing a 13 class classification problem
    dataset_path = Path("exercise_2/data/ShapeNetVox32")  # path to voxel data - make sure you've downloaded the ShapeNet voxel models to the correct path
    class_name_mapping = json.loads(Path("exercise_2/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        """
        :param split: one of 'train', 'val' or 'overfit' - for training, validation or overfitting split
        """
        super().__init__()
        assert split in ['train', 'val', 'overfit']

        self.items = Path(f"exercise_2/data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        """
        PyTorch requires you to provide a getitem implementation for your dataset.
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of data corresponding to the shape. In particular, this dictionary has keys
                 "name", given as "<shape_category>/<shape_identifier>",
                 "voxel", a 1x32x32x32 numpy float32 array representing the shape
                 "label", a number in [0, 12] representing the class of the shape
        """
        # TODO Get item associated with index, get class, load voxels with ShapeNetVox.get_shape_voxels
        item = self.items[index]
        # Hint: since shape names are in the format "<shape_class>/<shape_identifier>", the first part gives the class
        item_class = item.split("/")[0]
        # read voxels from binvox format on disk as 3d numpy arrays
        voxels = ShapeNetVox.get_shape_voxels(item)
        return {
            "name": item,
            "voxel": voxels[np.newaxis, :, :, :],  # we add an extra dimension as the channel axis, since pytorch 3d tensors are Batch x Channel x Depth x Height x Width
            "label": ShapeNetVox.classes.index(item_class)  # label is 0 indexed position in sorted class list, e.g. 02691156 is label 0, 02828884 is label 1 and so on.
        }

    def __len__(self):
        """
        :return: length of the dataset
        """
        # TODO Implement
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['voxel'] = batch['voxel'].to(device)
        batch['label'] = batch['label'].to(device)

    @staticmethod
    def get_shape_voxels(shapenet_id):
        """
        Utility method for reading a ShapeNet voxel grid from disk, reads voxels from binvox format on disk as 3d numpy arrays
        :param shapenet_id: Shape ID of the form <shape_class>/<shape_identifier>, e.g. 03001627/f913501826c588e89753496ba23f2183
        :return: a numpy array representing the shape voxels
        """
        with open(ShapeNetVox.dataset_path / shapenet_id / "model.binvox", "rb") as fptr:
            voxels = read_as_3d_array(fptr).astype(np.float32)
        return voxels


class ShapeNetPoints(torch.utils.data.Dataset):
    num_classes = 13  # we'll be performing a 13 class classification problem
    dataset_path = Path("exercise_2/data/ShapeNetPointClouds/")  # path to point cloud data
    class_name_mapping = json.loads(Path("exercise_2/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        # TODO Read sample IDs from the correct split file and store in self.items
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.items = Path(f"exercise_2/data/splits/shapenet/{split}.txt").read_text().splitlines()
        pass

    def __getitem__(self, index):
        # TODO Get item associated with index, get class, load points with ShapeNetPoints.get_point_cloud

        # Hint: Since shape names are in the format "<shape_class>/<shape_identifier>", the first part gives the class
        item = self.items[index]
        item_class = item.split("/")[0]
        points = ShapeNetPoints.get_point_cloud(item)

        return {
            "name": item,  # The item ID
            "points": points,
            "label": ShapeNetPoints.classes.index(item_class)  # Label is 0 indexed position in sorted class list, e.g. 02691156 is label 0, 02828884 is label 1 and so on.
        }

    def __len__(self):
        # TODO Implement
        return len(self.items)
        pass

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['points'] = batch['points'].to(device)
        batch['label'] = batch['label'].to(device)

    @staticmethod
    def get_point_cloud(shapenet_id):
        """
        Utility method for reading a ShapeNet point cloud from disk, reads points from obj files on disk as 3d numpy arrays
        :param shapenet_id: Shape ID of the form <shape_class>/<shape_identifier>, e.g. 03001627/f913501826c588e89753496ba23f2183
        :return: a numpy array representing the point cloud, in shape 3 x 1024
        """
        category_id, shape_id = shapenet_id.split('/')

        # TODO Implement
        points = trimesh.load(ShapeNetPoints.dataset_path/category_id/(shape_id+".obj")).vertices
        return np.array(points).T
        pass
