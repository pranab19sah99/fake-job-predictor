import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FakeJobClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_token = output.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_token))
        return logits


def load_bert_model(model_path: str):
    model = FakeJobClassifier()
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def load_tokenizer(tokenizer_path: str):
    return DistilBertTokenizerFast.from_pretrained(tokenizer_path)


@torch.no_grad()
def bert_predict(
    model,
    tokenizer,
    text: str,
    max_length: int = 256
):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    logits = model(input_ids, attention_mask)
    probs = torch.softmax(logits, dim=1)

    fake_prob = probs[0][1].item()
    return fake_prob
