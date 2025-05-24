# SDG Patent Multi-Label Classification (Production Pipeline)

## Files included:
- train.py              : Main training and evaluation script
- patent_sdg_dataset.py : Dataset, chunking, and loader functions
- collate_fn.py         : Batch collation for variable chunk sizes
- model.py              : Model loader
- utils.py              : Metrics, thresholding, aggregation
- predict.py            : Inference for new data

## Prerequisites:
- Python 3.8+ recommended
- pip install torch pandas scikit-learn tqdm transformers

## Data:
- Place your CSV.GZ patent data in a folder such as `../data/`
- Update DATA_PATH in train.py as needed

## Training:
```bash
python train.py
```
- Will auto-use your GPU.
- Trains, validates, tunes thresholds, and tests model. Model and thresholds saved to `./sdg_bert_model` and `sdg_thresholds.npy`.

## Inference:
See `predict.py` for a function to predict SDGs for new patent text.
Example:
```
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from predict import predict_sdg_labels

model = AutoModelForSequenceClassification.from_pretrained('./sdg_bert_model').to('cuda')
tokenizer = AutoTokenizer.from_pretrained('./sdg_bert_model')
thresholds = np.load('sdg_thresholds.npy')

sdgs = predict_sdg_labels(claims, abstract, description, model, tokenizer, thresholds)
print(sdgs)
```

## Advanced:
- Increase BATCH_SIZE if your GPU has free memory.
- For API, use FastAPI and wrap the `predict_sdg_labels` function.

---