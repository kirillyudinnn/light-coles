import torch 
import torch.nn as nn
import numpy as np

from typing import Dict

class EventEncoder(nn.Module):
    def __init__(
            self,
            config: Dict
    ):
        """
        Event encoder contains embedding layers for categorical features 
        and batch normalization layers for numerical features
        :param config: features configuration
        
        -- Example 
            {
                "cat_features" : {
                                    "cat_feat_1" : {"num_embs" : 4, "emb_dim" : 16}, ....
                                 }

                "num_features" : ["num_feat_1", ...]
            }

        """
        super().__init__()

        self.embeddings = nn.ModuleDict()
        for embedding_name, params in config["cat_features"]:
            self.embeddings[embedding_name] = nn.Embedding(
                num_embeddings=params["num_embs"],
                embedding_dim=params["emb_dim"]
            )

        self.scalers = nn.ModuleDict()
        for feature_name in config["num_features"]:
            self.scalers[feature_name] = CustomBatchNorm()


class CustomBatchNorm(nn.Module):
    """
        BatchNorm layer for numerical features without padded elements in batch
        to avoid biased parameters
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        pass
