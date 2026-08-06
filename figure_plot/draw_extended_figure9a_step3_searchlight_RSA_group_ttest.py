import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import fdrcorrection
import os


def perform_brain_ttest_analysis(main_data, use_fdr=True, alpha=0.05, uncorrected_threshold=0.001):
    n_subjects, n_vertices = main_data.shape
    
    mean_corrs = np.zeros(n_vertices)
    for vertex in range(n_vertices):
        vertex_data = main_data[:, vertex]
        valid_data = vertex_data[~np.isnan(vertex_data)]
        
        if len(valid_data) > 0:
            mean_corrs[vertex] = np.mean(valid_data)
        else:
            mean_corrs[vertex] = np.nan
            
    max_corr = np.nanmax(mean_corrs)

    t_stats = np.zeros(n_vertices)
    p_values = np.zeros(n_vertices)
    
    for vertex in range(n_vertices):
        vertex_data = main_data[:, vertex]
        valid_data = vertex_data[~np.isnan(vertex_data)]
        
        if len(valid_data) > 1:  
            t_stat, p_val = stats.ttest_1samp(valid_data, 0)
            t_stats[vertex] = t_stat
            p_values[vertex] = p_val
        else:
            t_stats[vertex] = np.nan
            p_values[vertex] = np.nan
    
    mean_corrs_threshold = mean_corrs.copy()
    
    if use_fdr:
        valid_mask = ~np.isnan(p_values)
        valid_p = p_values[valid_mask]
        
        if len(valid_p) > 0:
            rejected, adjusted_p = fdrcorrection(valid_p, alpha=alpha, method='indep')

            adj_h = np.zeros(n_vertices, dtype=bool)
            adj_p = np.full(n_vertices, np.nan)
            
            adj_h[valid_mask] = rejected
            adj_p[valid_mask] = adjusted_p

            mean_corrs_threshold[~adj_h] = np.nan
        else:
            adj_h = np.zeros(n_vertices, dtype=bool)
            adj_p = np.full(n_vertices, np.nan)
            mean_corrs_threshold[:] = np.nan
    else:
        adj_h = p_values <= uncorrected_threshold
        adj_p = p_values
        mean_corrs_threshold[p_values > uncorrected_threshold] = np.nan

    sig_positive = mean_corrs_threshold[mean_corrs_threshold > 0]
    sig_negative = mean_corrs_threshold[mean_corrs_threshold < 0]
    
    min_positive = np.nanmin(sig_positive) if len(sig_positive) > 0 else np.nan
    max_negative = np.nanmax(sig_negative) if len(sig_negative) > 0 else np.nan
    
    return {
        'mean_corrs': mean_corrs,
        'mean_corrs_threshold': mean_corrs_threshold,
        'max_corr': max_corr,
        't_stats': t_stats,
        'p_values': p_values,
        'adj_h': adj_h,
        'adj_p': adj_p,
        'min_positive_sig': min_positive,
        'max_negative_sig': max_negative,
        'n_significant': np.sum(adj_h),
        'n_positive_sig': np.sum(mean_corrs_threshold > 0),
        'n_negative_sig': np.sum(mean_corrs_threshold < 0)
    }

def perform_contrast_analysis(main_data, contrast_data, use_fdr=True, alpha=0.05, 
                            uncorrected_threshold=0.001, rectify_negative=False):
    if rectify_negative:
        main_data = main_data.copy()
        contrast_data = contrast_data.copy()
        main_data[main_data < 0] = 0
        contrast_data[contrast_data < 0] = 0
    
    n_subjects, n_vertices = main_data.shape

    difference_data = main_data - contrast_data
    mean_diff = np.nanmean(difference_data, axis=0)

    t_stats = np.zeros(n_vertices)
    p_values = np.zeros(n_vertices)
    
    for vertex in range(n_vertices):
        diff_vertex = difference_data[:, vertex]
        valid_diff = diff_vertex[~np.isnan(diff_vertex)]
        
        if len(valid_diff) > 1:
            t_stat, p_val = stats.ttest_1samp(valid_diff, 0)
            t_stats[vertex] = t_stat
            p_values[vertex] = p_val
        else:
            t_stats[vertex] = np.nan
            p_values[vertex] = np.nan

    mean_diff_threshold = mean_diff.copy()
    
    if use_fdr:
        valid_mask = ~np.isnan(p_values)
        valid_p = p_values[valid_mask]
        
        if len(valid_p) > 0:
            rejected, adjusted_p = fdrcorrection(valid_p, alpha=alpha, method='indep')
            
            adj_h = np.zeros(n_vertices, dtype=bool)
            adj_p = np.full(n_vertices, np.nan)
            
            adj_h[valid_mask] = rejected
            adj_p[valid_mask] = adjusted_p
            
            adj_h[np.isnan(adj_h)] = False
            
            mean_diff_threshold[~adj_h] = np.nan
        else:
            adj_h = np.zeros(n_vertices, dtype=bool)
            adj_p = np.full(n_vertices, np.nan)
            mean_diff_threshold[:] = np.nan
    else:
        adj_h = p_values <= uncorrected_threshold
        adj_p = p_values
        mean_diff_threshold[p_values > uncorrected_threshold] = np.nan

    sig_positive = mean_diff_threshold[mean_diff_threshold > 0]
    sig_negative = mean_diff_threshold[mean_diff_threshold < 0]
    
    min_positive = np.nanmin(sig_positive) if len(sig_positive) > 0 else np.nan
    max_negative = np.nanmax(sig_negative) if len(sig_negative) > 0 else np.nan
    
    return {
        'difference_data': difference_data,
        'mean_diff': mean_diff,
        'mean_diff_threshold': mean_diff_threshold,
        't_stats': t_stats,
        'p_values': p_values,
        'adj_h': adj_h,
        'adj_p': adj_p,
        'min_positive_sig': min_positive,
        'max_negative_sig': max_negative,
        'n_significant': np.sum(adj_h),
        'n_positive_sig': np.sum(mean_diff_threshold > 0),
        'n_negative_sig': np.sum(mean_diff_threshold < 0)
    }

def load_all_subjects_data(base_path, model_name, n_subjects=30):
    all_data = []
    print(f"--- Loading data for model: {model_name} ---")
    for i in range(1, n_subjects + 1):
        sub_id = f"sub-{i:02d}"
        file_path = os.path.join(base_path, sub_id, f"{sub_id}_selected_all_searchlight_results.npy")
        if i == 17:
            print("Skipping subject 17 as requested.")
            continue
        
        if os.path.exists(file_path):
            print(f"Loading {file_path}")
            data = np.load(file_path)
            all_data.append(data)
        else:
            print(f"Warning: File not found, skipping: {file_path}")
            pass

    if not all_data:
        print(f"Error: No data found for model {model_name}. Exiting.")
        return None
        
    return np.stack(all_data, axis=0)


if __name__ == "__main__":
    np.random.seed(42)
    main_model = 'MLLM_qwen_7B'
    base_data_path = f'data/pycortex/data/searchlight/{main_model}/'

    output_dir = f"data/pycortex/data/searchlight/{main_model}/result_ttest"
    os.makedirs(output_dir, exist_ok=True)
    main_data = load_all_subjects_data(base_data_path, main_model)
    
    if main_data is not None:
        results = perform_brain_ttest_analysis(main_data, use_fdr=True, alpha=0.05)
        mean_corrs = results['mean_corrs']
        significant_mask = results['adj_h']
        
        mean_corrs_fdr_masked = mean_corrs.copy()
        mean_corrs_fdr_masked[~significant_mask] = 0
        save_path = os.path.join(output_dir, f'{main_model}_mean_pearson_r_fdr_masked.npy')
        np.save(save_path, mean_corrs_fdr_masked)

        uncorrected_sig_count = np.nansum(results['p_values'] < 0.05)
        fdr_sig_count = results['n_significant']
