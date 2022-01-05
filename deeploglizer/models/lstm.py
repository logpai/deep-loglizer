from typing import Dict, Optional
import math
import torch
from torch import nn

from deeploglizer.models import ForecastBasedModel


class Attention(nn.Module):
    def __init__(self, input_size, max_seq_len):
        super(Attention, self).__init__()
        self.atten_w = nn.Parameter(torch.randn(max_seq_len, input_size, 1))
        self.atten_bias = nn.Parameter(torch.randn(max_seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.zeros(self.atten_bias)

    def forward(self, lstm_input):
        input_tensor = lstm_input.transpose(1, 0)  # f x b x d

        input_tensor = (
            torch.bmm(input_tensor, self.atten_w) + self.atten_bias
        )  # f x b x out
        input_tensor = input_tensor.transpose(1, 0)
        atten_weight = input_tensor.tanh()

        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), lstm_input).squeeze()

        return weighted_sum

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def zeros(self, tensor):
        if tensor is not None:
            tensor.data.fill_(0)


class LSTM(ForecastBasedModel):
    def __init__(
        self,
        meta_data:dict,
        hidden_size:int=100,
        num_directions:int=2,
        num_layers:int=1,
        window_size:Optional[int]=None,
        use_attention:bool=False,
        embedding_dim:int=16,
        model_save_path:str="./lstm_models",
        feature_type:str="sequentials",
        label_type:str="next_log",
        eval_type:str="session",
        topk:int=5,
        use_tfidf:bool=False,
        freeze:bool=False,
        gpu:int=-1,
        **kwargs
    ):

        super().__init__(
            meta_data=meta_data,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            eval_type=eval_type,
            topk=topk,
            use_tfidf=use_tfidf,
            embedding_dim=embedding_dim,
            freeze=freeze,
            gpu=gpu,
            **kwargs
        )

        num_labels = meta_data["num_labels"]
        self.feature_type = feature_type
        self.label_type = label_type
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.window_size = window_size
        self.use_attention = use_attention
        self.use_tfidf = use_tfidf
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=(self.num_directions == 2),
        )
        
        if self.use_attention:
            assert window_size is not None, "window size must be set if use attention"
            self.attn = Attention(hidden_size * num_directions, window_size)

        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.MSELoss()

        self.prediction_layer = nn.Linear(
            self.hidden_size * self.num_directions, num_labels
        )

    def forward(self, input_dict:dict) -> Dict[str, float]:
        if self.label_type == "anomaly":
            y = input_dict["window_anomalies"].long().view(-1)
        elif self.label_type == "next_log":
            y = input_dict["window_labels"].long().view(-1)
        self.batch_size = y.size()[0]
        x = input_dict["features"]
        x = self.embedder(x)

        if self.feature_type == "semantics":
            if not self.use_tfidf:
                x = x.sum(dim=-2)  # add tf-idf

        outputs, _ = self.lstm(x.float())

        if self.use_attention:
            representation = self.attn(outputs)
        else:
            # representation = outputs.mean(dim=1)
            representation = outputs[:, -1, :]

        logits = self.prediction_layer(representation)
        y_pred = logits.softmax(dim=-1)
        loss = self.criterion(logits, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        
        return return_dict
