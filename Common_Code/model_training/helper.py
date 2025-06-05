import numpy as np

def calculate_average_auc_scores(auc_scores_per_condition, sample_counts_per_condition):
    num_folds = len(list(auc_scores_per_condition.values())[0])
    average_auc_scores_per_fold = []

    for fold in range(num_folds):
        weighted_aucs = []

        for condition_type in auc_scores_per_condition.keys():
            auc_scores_fold = auc_scores_per_condition[condition_type][fold]
            sample_counts_fold = sample_counts_per_condition[condition_type][fold]

            weighted_auc = auc_scores_fold * sample_counts_fold
            #print(auc_scores_fold, f"sample_counts[{condition_type}][{fold}]", sample_counts_fold)
            weighted_aucs.append(weighted_auc)

        total_weighted_auc = np.sum(weighted_aucs)
        total_samples = np.sum([sample_counts_per_condition[condition_type][fold] for condition_type in auc_scores_per_condition.keys()])
        average_auc_score_fold = total_weighted_auc / total_samples
        average_auc_scores_per_fold.append(average_auc_score_fold)

    return average_auc_scores_per_fold