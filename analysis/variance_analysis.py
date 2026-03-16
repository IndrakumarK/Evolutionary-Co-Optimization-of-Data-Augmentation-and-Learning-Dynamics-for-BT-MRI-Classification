import numpy as np
from scipy.stats import ttest_rel


def aggregate_seeds(results):
    results = np.array(results)
    return results.mean(), results.std()


def paired_test(baseline, etdacvo):
    t_stat, p_value = ttest_rel(baseline, etdacvo)
    return t_stat, p_value