import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


def calculate_precision_recall(labels, logits, threshold):
    """Calculates precision and recall for a given threshold."""
    predicted_positive = (logits >= threshold).astype(int)
    true_positive = np.sum(labels * predicted_positive)
    false_positive = np.sum((1 - labels) * predicted_positive)
    false_negative = np.sum(labels * (1 - predicted_positive))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    return precision, recall

def calculate_aupr_trapezoidal(recall, precision):
    """Calculates AUPR using the trapezoidal rule."""
    aupr = 0.0
    # Sort by recall 
    #argsort is used to preserve the correspondent order of the precision
    sorted_indices = np.argsort(recall)
    sorted_recall = recall[sorted_indices]
    sorted_precision = precision[sorted_indices]

    for i in range(len(sorted_recall) - 1):
        base = sorted_recall[i + 1] - sorted_recall[i]
        avg_heigh = (sorted_precision[i] + sorted_precision[i + 1]) / 2.0
        aupr += base*avg_heigh
    return aupr

# MICRO-AUPR
# = flatten (labels & logits) -> one giant vector
# -> compute precision and recall over a range of thresholds
# -> apply trapezoidal rule on the entire PR curve
# => gives a single micro-AUPR value

# MACRO-AUPR
# = for each term (i.e., column):
#     - extract term[i] across all proteins
#     - compute precision and recall over a range of thresholds
#     - apply trapezoidal rule to get AUPR for this term
# -> average AUPRs across all terms
# => gives a single macro-AUPR value


def calculate_term_centric_aupr( labels,logits,  thresholds=np.linspace(0, 1, 100)):

    #columns
    num_terms = labels.shape[1]
    #rows
    num_proteins = labels.shape[0]

    # Micro-AUPR
    micro_precisions = []
    micro_recalls = []
    # one giant vector, all rows together
    flattened_labels = labels.flatten()
    flattened_logits = logits.flatten()
    
    for threshold in thresholds:
        precision, recall = calculate_precision_recall(flattened_labels, flattened_logits, threshold)
        micro_precisions.append(precision)
        micro_recalls.append(recall)
    micro_aupr = calculate_aupr_trapezoidal(np.array(micro_recalls), np.array(micro_precisions))

    # Macro-AUPR
    macro_auprs = []

    for term_index in range(num_terms):
        #particular term from every rows
        term_labels = labels[:, term_index]
        term_logits = logits[:, term_index]
        if np.sum(term_labels) == 0:
            continue 
        macro_precisions_term = []
        macro_recalls_term = []
        for threshold in thresholds:
            precision, recall = calculate_precision_recall(term_labels, term_logits, threshold)
            macro_precisions_term.append(precision)
            macro_recalls_term.append(recall)
        aupr_term = calculate_aupr_trapezoidal(np.array(macro_recalls_term), np.array(macro_precisions_term))
        macro_auprs.append(aupr_term)

    macro_aupr = np.mean(macro_auprs) if macro_auprs else 0.0

    return micro_aupr, macro_aupr

#over a range of thereshold
#calculate the f1 score for each protien
#extract the maximum f1 score across all the protiens.
#prcison are calculated for protiens that has atleast one single potisive example


def calculate_protein_centric_fmax( labels,logits):
    # t ∈ [0,1]
    thresholds=np.arange(0.0, 1.0, 0.01)
    num_proteins = labels.shape[0]
    f1_scores = []

    for threshold in thresholds:
        all_precisions = []
        all_recalls = []

        for i in range(num_proteins):
            #Ti
            true_terms = np.where(labels[i] == 1)[0]
            predicted_probs = logits[i]
            
            #Pi(t)
            predicted_terms = np.where(predicted_probs >= threshold)[0]
            # print(labels[i]," \n Lables ",true_terms," :terms \n lets see \n probs: " ,predicted_terms)
        
            #Ti ^ Pi(t)   np.isin -> and operation 
            tp = np.sum(np.isin(predicted_terms, true_terms))
            # if tp:
            #     print("tp is postivi for protien ",i," and for threshold ", thereshold)
            #tp+fp = len(predicted_terms)
            # fp = len(predicted_terms) - tp
            #tp+fn = len(true_terms)


            # Calculate precision for the current protein (only if at least one term is predicted)
            if len(predicted_terms) > 0:
                precision = tp / len(predicted_terms)
                all_precisions.append(precision)

            # Calculate recall for the current protein
            recall = tp / len(true_terms) if len(true_terms) > 0 else 0.0
            all_recalls.append(recall)

        # Calculate overall precision (averaged over proteins with at least one prediction)
        overall_precision = np.mean(all_precisions) if all_precisions else 0.0

        # Calculate overall recall (averaged over all proteins)
        overall_recall = np.mean(all_recalls) if all_recalls else 0.0

        # Calculate F1-score
        if (overall_precision + overall_recall) > 0:
            f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
            f1_scores.append(f1)
        else:
            f1_scores.append(0.0)
    
    print(len(f1_scores)," f1 is calculated over ",num_proteins, " samples")

    return np.max(f1_scores) if f1_scores else 0.0
