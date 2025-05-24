from transformers import AutoModelForSequenceClassification

def get_model(pretrained_name="anferico/bert-for-patents", num_labels=17):
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    return model
