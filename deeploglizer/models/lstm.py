from torch import nn
from deeploglizer.models import ForcastBasedModel


class LSTM(ForcastBasedModel):
    def __init__(
        self, num_labels, topk=5, hidden_size=100, num_directions=2, device=-1
    ):
        super().__init__(topk=topk, device=device)
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=(self.num_directions == 2),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.prediction_layer = nn.Linear(
            self.hidden_size * self.num_directions, num_labels + 1
        )

    def forward(self, input_dict):
        y = input_dict["window_labels"].long().view(-1).to(self.device)
        self.batch_size = y.size()[0]
        x = input_dict["features"].view(self.batch_size, -1, 1).to(self.device)
        outputs, hidden = self.rnn(x.float())
        representation = outputs.sum(dim=1)
        logits = self.prediction_layer(representation)
        y_pred = logits.softmax(dim=-1)
        loss = self.criterion(logits, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict