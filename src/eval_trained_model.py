import spacy
from sklearn.metrics import classification_report, precision_recall_fscore_support
from collections import defaultdict
from custom_train_model import verify_entity_alignment, adjust_entity_offsets
import utils.config as config
import utils.data as data
import time


def evaluate_model(model, test_data):
    true_entities = []
    pred_entities = []
    entity_counts = defaultdict(int)
    skipped_count = 0  # Keep track of skipped examples

    for text, annotations in test_data:
        entities = annotations.get("entities", [])

        if not verify_entity_alignment(model, text, entities):
            print(f"Adjusting entities for evaluation: {text}")
            adjusted_entities = adjust_entity_offsets(model, text, entities)
            if verify_entity_alignment(model, text, adjusted_entities):
                annotations["entities"] = adjusted_entities
            else:
                print(f"Could not fix alignment for evaluation: {text}")
                skipped_count += 1
                continue  # Skip to the next example

        true_ents = annotations.get("entities", [])
        true_entities.extend([(text[start:end], label) for start, end, label in true_ents])

        # Inference
        start_time_inference = time.time()
        doc = model(text)
        end_time_inference = time.time()
        inference_time = end_time_inference - start_time_inference

        pred_ents = [(ent.text, ent.label_) for ent in doc.ents]
        pred_entities.extend(pred_ents)

        for start, end, label in true_ents:
            entity_counts[label] += 1

    print("Entity distribution in test data:")
    for label, count in entity_counts.items():
        print(f"{label}: {count}")

    print(f"\nNumber of examples skipped due to alignment issues: {skipped_count}") # Report skipped examples

    # Correct Matching Logic:
    matched_predictions = []
    for true_text, true_label in true_entities:
        found_match = False
        for pred_text, pred_label in pred_entities:
            if true_text == pred_text and true_label == pred_label:  # Exact match
                matched_predictions.append(pred_label)
                found_match = True
                break  # Important: Avoid double-counting matches
        if not found_match:
            matched_predictions.append("O") #append 'O' to represent that no matching entity was found in the predictions.

    print("\nClassification Report:")
    y_true = [ent[1] for ent in true_entities]
    print(classification_report(y_true, matched_predictions, zero_division=0)) 

    # Get unique labels (as strings) for metrics calculation
    labels = sorted(list(set(y_true + matched_predictions)))

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, matched_predictions, labels=labels, average=None)

    print("\nPer-Entity Metrics:")
    for label, p, r, f in zip(labels, precision, recall, f1):
        print(f"{label}: Precision={p:.2f}, Recall={r:.2f}, F1-Score={f:.2f}")

    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(y_true, matched_predictions, average='weighted')
    print(f"\nOverall Metrics: Precision={overall_precision:.2f}, Recall={overall_recall:.2f}, F1-Score={overall_f1:.2f}")

    # Calculate overall accuracy
    correct_predictions = sum(1 for true_label, pred_label in zip(y_true, matched_predictions) if true_label == pred_label)
    overall_accuracy = correct_predictions / len(y_true) if y_true else 0.0 
    print(f"Overall Accuracy: {overall_accuracy:.2f}")

    # Total inference time
    inference_time
    print(f"Total Inference Time: {inference_time:.3f}secs")


if __name__ == '__main__':
    # Load the trained model
    trained_model = spacy.load(config.TRAINED_MODEL_NAME)
    
    # Evaluate the model on test data
    evaluate_model(trained_model, data.test_data)