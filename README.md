# Custom NER Model
Train a custom NER model with SpaCy

## Setup and Run
1. Clone the Repo
```bash
git clone https://github.com/pirevi/custom-ner-model.git
cd custom-ner-model
```

2. Create virtual environment and install all dependencies
```bash
python -m venv .venv
# After activating the virtual environment do ->
pip install -r requirements.txt
```
3. Update `src/utils/data.py` with train and test data

4. Configure `src/utils/config.py` file according to your needs

5. Run `src/custom_train_model.py`

6. Run `src/val_trained_model.py` to know model performance