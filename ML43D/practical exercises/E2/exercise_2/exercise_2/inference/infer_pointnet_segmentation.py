import torch

from exercise_2.data.shapenet_parts import ShapeNetParts
from exercise_2.model.pointnet import PointNetSegmentation


class InferenceHandlerPointNetSegmentation:
    """Utility for segmentation inference using trained PointNet network"""

    def __init__(self, ckpt):
        """
        :param ckpt: checkpoint path to weights of the trained network
        """
        self.model = PointNetSegmentation(ShapeNetParts.num_classes)
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()

    def infer_single(self, points):
        """
        Infer class of the shape given its point cloud representation
        :param points: points of shape 3 x 1024
        :return: part segmentation labels for the point cloud, as predicted by the model
        """
        input_tensor = torch.from_numpy(points).float().unsqueeze(0)
        prediction = torch.argmax(self.model(input_tensor)[0], dim=1)
        return prediction
