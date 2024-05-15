def find_best_transformers(model: any, transformer: str, best_transformers: dict[str, str],
                           best_correlations: dict[str, float]) -> tuple[dict[str, str], dict[str, float]]:
    corr_train = model.evaluate('Train')
    corr_dev = model.evaluate('Dev')
    corr_test = model.evaluate('Test')

    if corr_train > best_correlations['Train']:
        best_correlations['Train'] = corr_train
        best_transformers['Train'] = transformer
    if corr_dev > best_correlations['Dev']:
        best_correlations['Dev'] = corr_train
        best_transformers['Dev'] = transformer
    if corr_test > best_correlations['Test']:
        best_correlations['Test'] = corr_train
        best_transformers['Test'] = transformer

    return best_transformers, best_correlations
