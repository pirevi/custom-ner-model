import random
import spacy
import utils.config as config
import utils.data as data
from spacy.util import minibatch
from spacy.training.example import Example
from spacy.training import offsets_to_biluo_tags


def verify_entity_alignment(model, text, entities):
    """
    Function to verify entity alignment
    """
    doc = model.make_doc(text)
    tags = offsets_to_biluo_tags(doc, entities)
    if '-' in tags:
        print(f"Misaligned entities in: {text}")
        print(f"BILUO tags: {tags}")
        return False
    return True


def adjust_entity_offsets(model, text, entities):
    """
    Function to fix common alignment issues
    """
    doc = model.make_doc(text)
    adjusted_entities = []
    
    for start, end, label in entities:
        # Find closest token boundaries
        token_start = None
        token_end = None
        
        # Find the token that encompasses or is closest to the start
        for token in doc:
            if token.idx <= start < token.idx + len(token.text):
                token_start = token.idx
                break
        
        # Find the token that encompasses or is closest to the end
        for token in doc:
            if token.idx <= end <= token.idx + len(token.text):
                token_end = token.idx + len(token.text)
                break
        
        if token_start is not None and token_end is not None:
            adjusted_entities.append((token_start, token_end, label))
    
    # Verify the adjusted entities
    if verify_entity_alignment(model, text, adjusted_entities):
        return adjusted_entities
    return entities


def train_custom_ner_cnn_model(model, train_data, epochs=50, mini_batch_size=5, dropout=0.3):
    """
    Train a custom NER model using spaCy's cnn-based 'en_core_web_sm/md/lg' model.
    """
    # Prepare and verify training data
    verified_training_data = []
    for text, annotations in train_data:
        entities = annotations['entities']
        if not verify_entity_alignment(model, text, entities):
            print(f"Adjusting entities for: {text}")
            adjusted_entities = adjust_entity_offsets(model, text, entities)
            if verify_entity_alignment(model, text, adjusted_entities):
                verified_training_data.append((text, {"entities": adjusted_entities}))
            else:
                print(f"Could not fix alignment for: {text}")
        else:
            verified_training_data.append((text, annotations))

    # Set up the NER pipeline
    if 'ner' not in model.pipe_names:
        ner = model.add_pipe('ner', last=True)
    else:
        ner = model.get_pipe('ner')

    # Add labels
    labels = set()
    for _, annotations in verified_training_data:
        for ent in annotations['entities']:
            labels.add(ent[2])
            ner.add_label(ent[2])
    print("Training with following labels:", sorted(labels))

    # Training
    other_pipes = [pipe for pipe in model.pipe_names if pipe != 'ner']
    with model.disable_pipes(*other_pipes):
        optimizer = model.begin_training()
        
        for epoch in range(epochs):
            random.shuffle(verified_training_data)
            losses = {}
            batches = minibatch(verified_training_data, size=mini_batch_size)
            
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = model.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)

                model.update(examples, 
                             sgd=optimizer, # actually uses Adam by default
                             drop=dropout, 
                             losses=losses) 
                
            if (epoch + 1) % config.LOSS_DISPLAY_RATE == 0:
                print(f"Epoch {epoch+1}, Losses: {losses}")

    # Save the model
    model.to_disk(config.TRAINED_MODEL_NAME)


def train_custom_ner_transformer_model(model, train_data, epochs=50, mini_batch_size=5, dropout=0.3):
    """
    Train a custom NER model using spaCy's transformer-based 'en_core_web_trf' model.
    """
    # Prepare and verify training data
    verified_training_data = []
    for text, annotations in train_data:
        entities = annotations['entities']
        if not verify_entity_alignment(model, text, entities):
            print(f"Adjusting entities for: {text}")
            adjusted_entities = adjust_entity_offsets(model, text, entities)
            if verify_entity_alignment(model, text, adjusted_entities):
                verified_training_data.append((text, {"entities": adjusted_entities}))
            else:
                print(f"Could not fix alignment for: {text}")
        else:
            verified_training_data.append((text, annotations))

    # Debug information
    print("Model pipeline:", model.pipe_names)

    # Ensure NER component is present
    if "ner" not in model.pipe_names:
        ner = model.add_pipe("ner", last=True)
    else:
        ner = model.get_pipe("ner")

    # Add labels
    labels = set()
    for _, annotations in verified_training_data:
        for ent in annotations.get("entities", []):
            labels.add(ent[2])
            ner.add_label(ent[2])
    print("Training with following labels:", sorted(labels))

    # Convert to examples
    examples = []
    for text, annotations in verified_training_data:
        doc = model.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    print(f"Created {len(examples)} valid training examples")

    # Training
    other_pipes = [pipe for pipe in model.pipe_names if pipe not in ["ner", "transformer"]]
    with model.disable_pipes(*other_pipes):
        try:
            print("Starting training...")
            optimizer = model.resume_training()
            
            for epoch in range(epochs):
                random.shuffle(examples)
                losses = {}
                
                # Process in batches
                batches = minibatch(examples, size=mini_batch_size)
                for batch_id, batch in enumerate(batches):
                    try:
                        # print(f"Processing batch {batch_id + 1}")
                        model.update(
                            batch,
                            drop=dropout,
                            losses=losses,
                            sgd=optimizer
                        )
                        
                    except Exception as e:
                        print(f"Error during batch {batch_id + 1} update:")
                        print(f"Error message: {str(e)}")

                if (epoch + 1) % config.LOSS_DISPLAY_RATE == 0:
                    print(f"Epoch {epoch + 1}, Losses: {losses}")

        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    model.to_disk(config.TRAINED_MODEL_NAME)


if __name__ == '__main__':
    # Enable GPU if available and required
    if spacy.prefer_gpu() and config.PREFER_GPU:
        print("Using GPU for training.")
    else:
        spacy.require_cpu()
        print("Using CPU for training.")

    # Load base spaCy model
    try:
        nlp_base_model = spacy.load(config.BASE_NLP_MODEL)
    except OSError:
        import sys
        import subprocess
        print(f"Model {config.BASE_NLP_MODEL} not found!")
        valid_models = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"]
        if config.BASE_NLP_MODEL not in valid_models:
            print(f"Invalid model. Only following models allowed -> {valid_models}")
        else:
            print(f"Installing model: {config.BASE_NLP_MODEL}...")
            subprocess.run([sys.executable, "-m", "spacy", "download", config.BASE_NLP_MODEL], check=True)
            nlp_base_model = spacy.load(config.BASE_NLP_MODEL)
    except Exception:
        raise RuntimeError("Failed to load spaCy model!")

    # Start training
    if config.BASE_NLP_MODEL != "en_core_web_trf":
        train_custom_ner_cnn_model(
            nlp_base_model, data.train_data, 
            epochs=config.EPOCHS, mini_batch_size=config.MINI_BATCH_SIZE, dropout=config.DROPOUT)
    else:
        train_custom_ner_transformer_model(
            nlp_base_model, data.train_data, 
            epochs=config.EPOCHS, mini_batch_size=config.MINI_BATCH_SIZE, dropout=config.DROPOUT)
        