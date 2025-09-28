
def make_data_registry(data_mgr):
    return {
        "dataset_stats":                  lambda variant: data_mgr.generate_dataset_stats(variant),
        "entity_tag_distribution":        lambda variant: data_mgr.generate_entity_tag_distribution(variant),
        "entity_span_distribution":       lambda variant: data_mgr.generate_entity_span_distribution(variant), 
        "entity_span_complexity":         lambda variant: data_mgr.generate_entity_span_complexity_distribution(variant), 
        "entity_tag_type_distribution":   lambda variant: data_mgr.generate_entity_tag_type_distribution(variant), 
        "type_to_word_ratio":             lambda variant: data_mgr.generate_type_to_word_ratio(variant),
        "word_type_frequency_distribution": lambda variant: data_mgr.generate_word_type_frequency_distribution(variant),
        "dataset_oov_rate":               lambda variant: data_mgr.generate_dataset_oov_rate(variant),
        "entity_tag_oov_rate":            lambda variant: data_mgr.generate_entity_tag_oov_rate(variant),
        "tokenised_dataset_stats":        lambda variant: data_mgr.generate_tokenised_dataset_stats(variant),
        "entity_tag_token_type_distribution": lambda variant: data_mgr.generate_entity_tag_token_type_distribution(variant),
        "type_to_token_ratio":            lambda variant: data_mgr.generate_type_to_token_ratio(variant),
        "token_type_frequency_distribution":  lambda variant: data_mgr.generate_token_type_frequency_distribution(variant),
        "entity_tag_token_oov_rate":      lambda variant: data_mgr.generate_entity_tag_token_oov_rate(variant),
        "word_type_overlap_train":        lambda variant: data_mgr.generate_word_type_overlap_train(variant),
        "word_type_overlap_test":         lambda variant: data_mgr.generate_word_type_overlap_test(variant),
        "token_type_overlap_train":       lambda variant: data_mgr.generate_token_type_overlap_train(variant),
        "token_type_overlap_test":        lambda variant: data_mgr.generate_token_type_overlap_test(variant),
        "tokenization_rate":              lambda variant: data_mgr.generate_tokenization_rate(variant),
        "ambiguity":                      lambda variant: data_mgr.generate_ambiguity(variant),
        "consistency_metrics":            lambda variant: data_mgr.generate_consistency_metrics(variant),
    }

def make_model_registry(model_mgr):
    return {
        "loss": lambda variant: model_mgr.generate_loss(variant),
        "silhouette":  lambda variant: model_mgr.generate_silhouette(variant),
        "prediction_uncertainty":  lambda variant: model_mgr.generate_prediction_uncertainty(variant),
        "per_class_confidence": lambda variant: model_mgr.generate_pre_class_confidence(variant),
        "token_confidence": lambda variant: model_mgr.generate_token_confidence(variant),
        "confidence_confusion": lambda variant: model_mgr.generate_confidence_confusion(variant),
    }


def make_evaluation_registry(eval_mgr):
    return {
        "overall_token_vs_entity":          lambda variant: eval_mgr.generate_overall_token_vs_entity(variant),
        "overall_span_schemes":             lambda variant: eval_mgr.generate_overall_entity_schemes(variant),
        "span_f1":                          lambda variant: eval_mgr.generate_span_f1(variant),
        "span_precision_recall":            lambda variant: eval_mgr.generate_span_precision_recall(variant),
        "span_support":                     lambda variant: eval_mgr.generate_span_support(variant),
        "span_prediction_outcomes":         lambda variant: eval_mgr.generate_span_prediction_outcomes(variant),
        "span_prediction_errors":           lambda variant: eval_mgr.generate_span_prediction_errors(variant),
        "span_error_types":                 lambda variant: eval_mgr.generate_span_error_types(variant),
        "span_error_types_fp":              lambda variant: eval_mgr.generate_span_error_types_fp(variant),
        "span_error_types_fn":              lambda variant: eval_mgr.generate_span_error_types_fn(variant),
        "span_entity_errors_fp_heatmap":    lambda variant: eval_mgr.generate_span_entity_errors_fp(variant),
        "span_entity_errors_fn_heatmap":    lambda variant: eval_mgr.generate_span_entity_errors_fn(variant),
        "token_f1":                         lambda variant: eval_mgr.generate_token_f1(variant),
        "token_precision_recall":           lambda variant: eval_mgr.generate_token_pr_rc(variant),
        "token_support":                    lambda variant: eval_mgr.generate_token_support(variant),
        "token_prediction_outcomes":        lambda variant: eval_mgr.generate_token_prediction_outcomes(variant),
        "token_confusion_heatmap":          lambda variant: eval_mgr.generate_token_confusion_heatmap(variant),
        "token_support_correlations":       lambda variant: eval_mgr.generate_token_support_correlations(variant),
        "token_support_scatter":            lambda variant: eval_mgr.generate_token_support_scatter(variant),
        "token_spearman":                   lambda variant: eval_mgr.generate_token_spearman(variant),
        # "token_support_scatter_ar":         lambda variant: eval_mgr.generate_token_support_scatter_ar(variant),
        # "token_support_scatter_en":         lambda variant: eval_mgr.generate_token_support_scatter_en(variant),
        # "token_spearman_ar":                lambda variant: eval_mgr.generate_token_spearman_ar(variant),
        # "token_spearman_en":                lambda variant: eval_mgr.generate_token_spearman_en(variant),
    }