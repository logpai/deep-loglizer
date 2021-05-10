import torch
from torch import nn
from deeploglizer.models import ForcastBasedModel

from IPython import embed


class Transformer(ForcastBasedModel):
    def __init__(
        self,
        meta_data,
        embedding_dim=16,
        nhead=4,
        hidden_size=100,
        num_layers=1,
        feature_type="sequentials",
        topk=5,
        use_tfidf=False,
        pretrain_matrix=None,
        freeze=False,
        device=-1,
    ):
        super().__init__(
            meta_data=meta_data,
            feature_type=feature_type,
            topk=topk,
            use_tfidf=use_tfidf,
            embedding_dim=embedding_dim,
            pretrain_matrix=pretrain_matrix,
            freeze=freeze,
            device=device,
        )
        num_labels = meta_data["num_labels"]
        self.feature_type = feature_type
        self.hidden_size = hidden_size
        self.use_tfidf = use_tfidf

        self.cls = torch.zeros(1, 1, embedding_dim).to(self.device)
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.criterion = nn.CrossEntropyLoss()
        self.prediction_layer = nn.Linear(self.hidden_size * nhead, num_labels)

    def forward(self, input_dict):
        y = input_dict["window_labels"].long().view(-1)
        self.batch_size = y.size()[0]
        x = input_dict["features"]
        x = self.embedder(x)

        if self.feature_type == "semantics":
            if not self.use_tfidf:
                x = x.sum(dim=-2)  # add tf-idf
        x_transformed = self.transformer_encoder(x.transpose(1, 0))
        embed()
        outputs, hidden = self.transformer_encoder(x)
        representation = outputs.sum(dim=1)
        logits = self.prediction_layer(representation)
        y_pred = logits.softmax(dim=-1)
        loss = self.criterion(logits, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict