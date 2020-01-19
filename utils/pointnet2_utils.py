import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
from linalg_utils import pdist2, PDist2Order
from collections import namedtuple
import pytorch_utils as pt_utils
from typing import List, Tuple

import torch_points as tp


class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of points to gather in the ball
    """

    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(
        self,
        xyz: torch.Tensor,
        new_xyz: torch.Tensor,
        features: torch.Tensor = None,
        fps_idx: torch.IntTensor = None,
    ) -> Tuple[torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = tp.ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = tp.grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        raw_grouped_xyz = grouped_xyz
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = tp.grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [raw_grouped_xyz, grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3 + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = torch.cat([raw_grouped_xyz, grouped_xyz], dim=1)

        return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(
        self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: torch.Tensor = None
    ) -> Tuple[torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
