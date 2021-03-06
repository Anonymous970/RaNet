from torch import nn
import torch
import math
import numpy as np

from core.config import config
import models.frame_modules as frame_modules
import models.graph_modules as graph_modules
import models.prop_modules as prop_modules
import models.fusion_modules as fusion_modules
import models.map_modules as map_modules


class TAN(nn.Module):
    def __init__(self):
        super(TAN, self).__init__()

        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.graph_layer = getattr(graph_modules, config.TAN.GRAPH_MODULE.NAME)(config.TAN.GRAPH_MODULE.PARAMS)
        self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS)
        self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS)
        self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
        self.pred_layer = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)

    def forward(self, textual_input, textual_mask, visual_input):

        vis_h = self.frame_layer(visual_input.transpose(1, 2)) 
        vis_h = self.graph_layer(vis_h) 
        map_h, map_mask = self.prop_layer(vis_h) 
        fused_h = self.fusion_layer(textual_input, textual_mask, map_h)  
        fused_h = self.map_layer(fused_h, map_mask)  
        prediction = self.pred_layer(fused_h) * map_mask.float()  

        return prediction, map_mask


