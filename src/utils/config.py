# Model names
# Available Base Models: 'en_core_web_sm' (12MB) < 'en_core_web_md' (31MB) < 'en_core_web_lg' (382MB) < 'en_core_web_trf' (432MB)
BASE_NLP_MODEL = 'en_core_web_sm' 
TRAINED_MODEL_NAME = 'custom_model_ner_sm'
PREFER_GPU = False 

# Hyperparameters
EPOCHS = 50
LOSS_DISPLAY_RATE = 10 # display training loss in multiples
MINI_BATCH_SIZE = 5
DROPOUT = 0.3

