import torch
import torch.nn as nn
from person_pair import person_pair
from torchvision.models import resnet50
from torch_geometric.nn import GatedGraphConv

SIZE = 512
dim = 512
heads = 8
dim_head = 64
dropout = 0.
depth = 12
mlp_dim = 512


class groupGraph(nn.Module):
    def __init__(self, num_classes=6):
        super(groupGraph, self).__init__()

        self.person_pair = person_pair(num_classes)

        self.group = resnet50()
        self.group.fc = nn.Sequential(nn.Linear(self.group.fc.in_features, SIZE),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout())

        self.convGroup = GatedGraphConv(SIZE, num_layers=3)

        self.sigmoid = nn.Sigmoid()
        self.fc_groupClass = nn.Linear(SIZE, num_classes)

    def forward(self, pair, person_a, person_b, bbox, full_im, img_rel_num, edge_index):

        _, personPair, _, _, _, _ = self.person_pair(pair, person_a, person_b, bbox)

        group_feature = self.group(full_im)

        input_graph = torch.cat((personPair[0:img_rel_num[0]], group_feature[0].unsqueeze(dim=0)), dim=0)
        count = int(img_rel_num[0])
        for rel_num, feature in zip(img_rel_num[1:], group_feature[1:]):
            subgraph = torch.cat((personPair[count: (count + rel_num)], feature.unsqueeze(dim=0)), dim=0)
            input_graph = torch.cat((input_graph, subgraph), dim=0)
            count += rel_num

        result = self.convGroup(input_graph, edge_index)
        result = self.sigmoid(result)

        if img_rel_num[0] == 1:
            result_filter = result[0].unsqueeze(dim=0)
        else:
            result_filter = result[0:img_rel_num[0]]

        count = img_rel_num[0] + 1
        for num in img_rel_num[1:]:
            if num == 1:
                result_filter = torch.cat((result_filter, result[count].unsqueeze(dim=0)), dim=0)
            else:
                result_filter = torch.cat((result_filter, result[count:(count + num)]), dim=0)
            count += (num + 1)

        fc_groupClass = self.fc_groupClass(result_filter)

        return fc_groupClass, group_feature, result_filter
