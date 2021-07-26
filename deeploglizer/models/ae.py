import torch
import torch.nn.functional as F
from torch import nn

from deeploglizer.models import ForcastBasedModel


class AutoEncoder(ForcastBasedModel):
    def __init__(
        self,
        meta_data,
        hidden_size=100,
        num_layers=1,
        num_directions=2,
        embedding_dim=16,
        model_save_path="./ae_models",
        feature_type="sequentials",
        label_type="none",
        eval_type="session",
        topk=5,
        use_tfidf=False,
        freeze=False,
        gpu=-1,
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
        self.feature_type = feature_type
        self.label_type = label_type
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.use_tfidf = use_tfidf
        self.embedding_dim = embedding_dim
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=(self.num_directions == 2),
        )

        self.encoder = nn.Linear(
            self.hidden_size * self.num_directions, self.hidden_size // 2
        )

        self.decoder = nn.Linear(
            self.hidden_size // 2, self.hidden_size * self.num_directions
        )
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, input_dict):
        x = input_dict["features"]
        self.batch_size = x.size()[0]
        if self.embedding_dim == 1:
            x = x.unsqueeze(-1)
        else:
            x = self.embedder(x)

        if self.feature_type == "semantics":
            if not self.use_tfidf:
                x = x.sum(dim=-2)  # add tf-idf

        outputs, hidden = self.rnn(x.float())
        # representation = outputs.mean(dim=1)
        representation = outputs[:, -1, :]

        x_internal = self.encoder(representation)
        x_recst = self.decoder(x_internal)

        pred = self.criterion(x_recst, representation).mean(dim=-1)
        loss = pred.mean()
        return_dict = {"loss": loss, "y_pred": pred}
        return return_dict
