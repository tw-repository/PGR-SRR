import torch
import torch.nn as nn
from person_pair import person_pair
from torch_geometric.nn import GatedGraphConv

SIZE = 512
dim = 512
heads = 8
dim_head = 64
dropout = 0.
depth = 12
mlp_dim = 512


class pairGraph(nn.Module):
    def __init__(self, num_classes):
        super(pairGraph, self).__init__()

        self.person_pair = person_pair(num_classes)

        self.fc_person = nn.Linear(SIZE * 2, SIZE)

        self.convPair = GatedGraphConv(SIZE, num_layers=3)

        self.fc_pairClass = nn.Linear(SIZE, num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, pair, person_a, person_b, bbox, img_rel_num, edge_index):

        _, personPair, _, personA, personB, hLevelF = self.person_pair(pair, person_a, person_b, bbox)

        fc_person = self.fc_person(torch.cat((personA, personB), 1))

        input_graph = torch.cat((personPair[0:img_rel_num[0]], fc_person[0].unsqueeze(dim=0)), dim=0)
        count = int(img_rel_num[0])
        for rel_num, feature in zip(img_rel_num[1:], fc_person[1:]):
            subgraph = torch.cat((personPair[count: (count + rel_num)], feature.unsqueeze(dim=0)), dim=0)
            input_graph = torch.cat((input_graph, subgraph), dim=0)
            count += rel_num

        result = self.convPair(input_graph, edge_index)
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

        fc_pairClass = self.fc_pairClass(result_filter)

        return fc_pairClass, personPair, personA, personB, hLevelF, result_filter
