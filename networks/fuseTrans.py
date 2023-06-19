import torch
import torch.nn as nn
import numpy as np
from pairGraph import pairGraph
from groupGraph import groupGraph
from einops import repeat
from transformer import Transformer

SIZE = 512
dim = 512
heads = 8
dim_head = 64
dropout = 0.
depth = 12
mlp_dim = 512


class fusion(nn.Module):
    def __init__(self, num_classes=6):
        super(fusion, self).__init__()

        self.pairGraph = pairGraph(num_classes)
        self.groupGraph = groupGraph(num_classes)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))

        self.multiattn_intra = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.multiattn_inter = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.fc_Class = nn.Linear(SIZE, num_classes)
        self.ReLU = nn.ReLU()

    def forward(self, pair, person_a, person_b, bbox, full_im, img_rel_num, edge_index):

        _, personP, personA, personB, hLevelF, pair_graph = self.pairGraph(pair, person_a, person_b, bbox, img_rel_num, edge_index)
        _, group_feature, group_graph = self.groupGraph(pair, person_a, person_b, bbox, full_im, img_rel_num, edge_index)

        scene_new = group_feature[0].repeat(img_rel_num[0], 1)
        for i, num in enumerate(img_rel_num[1:]):
            scene_new = torch.cat([scene_new, group_feature[i + 1].repeat(num, 1)], dim=0)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=personP.shape[0])

        x1 = personP[:, np.newaxis, :]
        x5 = pair_graph[:, np.newaxis, :]
        x6 = group_graph[:, np.newaxis, :]
        x7 = scene_new[:, np.newaxis, :]

        fea_attn = torch.cat((cls_tokens, x1, x5, x6, x7), dim=-2)

        fea_mhsa = self.multiattn_intra(fea_attn)
        fea_mhsa = self.ReLU(fea_mhsa)

        cls_attn = fea_mhsa[:, 0, :]

        rel_num_1 = img_rel_num[0]
        img_inter_1 = cls_attn[0:rel_num_1].unsqueeze(0)
        output = self.multiattn_inter(img_inter_1).squeeze(0)
        count = int(rel_num_1)
        for rel_num in img_rel_num[1:]:
            if rel_num == 1:
                test_cls = cls_attn[count].unsqueeze(dim=0)
                output = torch.cat((output, test_cls), dim=0)
            else:
                img_inter = cls_attn[count:(count + rel_num)].unsqueeze(0)
                output_1 = self.multiattn_inter(img_inter).squeeze(0)
                output = torch.cat((output, output_1), dim=0)
            count += rel_num

        output = self.ReLU(cls_attn)
        final_result = self.fc_Class(output)

        return final_result
