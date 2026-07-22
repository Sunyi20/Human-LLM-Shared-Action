clear all; close all; clc;
base_dir = pwd;
addpath(base_dir)
addpath(genpath(fullfile(base_dir, 'helper_functions')))

index_file_path = 'data/folder_list/folder_list_full_sample_41.txt';
T_indices = readtable(index_file_path, 'FileType', 'text', 'ReadVariableNames', false, 'Format', '%d %[^\n]');
subset_indices_41 = T_indices.Var1 + 1;

wordposition48_path = fullfile('data', 'folder_list/wordposition48.txt');
wordposition48 = load(wordposition48_path);

if ~ismember(numel(wordposition48), [41, 48])
    candidate_wordposition48_paths = { ...
        fullfile('Human_Action', 'full_sample', 'wordposition48.txt'), ...
        fullfile('Human_Action', 'data', 'wordposition48.txt')};

    found_wordposition48 = false;
    for p = 1:numel(candidate_wordposition48_paths)
        candidate = candidate_wordposition48_paths{p};
        if exist(candidate, 'file')
            candidate_wordposition48 = load(candidate);
            if ismember(numel(candidate_wordposition48), [41, 48])
                warning('data/wordposition48.txt contains %d entries; using %s for the 355-object mapping.', ...
                    numel(wordposition48), candidate);
                wordposition48 = candidate_wordposition48;
                found_wordposition48 = true;
                break;
            end
        end
    end

    if ~found_wordposition48
        error('wordposition48 must contain 41 or 48 entries for the model mapping, but %s contains %d entries.', ...
            wordposition48_path, numel(wordposition48));
    end
end

wordposition48 = wordposition48(:) + 1;
subset_indices_41 = subset_indices_41(:);

assert(ismember(numel(wordposition48), [41, 48]), 'wordposition48 must contain 41 or 48 entries.');
assert(numel(subset_indices_41) == 41, 'subset_indices_41 must contain exactly 41 entries.');


%% ===================== 1. LLM qwen 7B =====================
fprintf('Computing LLM qwen 7B...\n');
dd = 'data/LLM_qwen_7B';
emb_all = [load(fullfile(dd, 'qwen_7B_spose_embedding_sorted_final.txt')), ...
            load(fullfile(dd, 'qwen_7B_spose_embedding_sorted_delete.txt'))];
load(fullfile(dd, 'RDM_48_LLM_qwen_7B.mat'));
r_values_llm7b = compute_rsm_curve_model(emb_all, wordposition48, subset_indices_41, RDM_triplet);
dims_llm7b = 1:length(r_values_llm7b);
fprintf('  Done. Plateau R = %.4f, dims = %d\n', r_values_llm7b(end), length(r_values_llm7b));

%% ===================== 2. LLM deepseek =====================
fprintf('Computing LLM deepseek...\n');
dd = 'data/LLM_deepseek';
emb_all = [load(fullfile(dd, 'deepseek_spose_embedding_sorted_final.txt')), ...
            load(fullfile(dd, 'deepseek_spose_embedding_sorted_delete.txt'))]; 
clear RDM_triplet RDM48_triplet;
load(fullfile(dd, 'RDM_48_LLM_deepseek.mat'));
if exist('RDM48_triplet', 'var'), RDM_use = RDM48_triplet;
elseif exist('RDM_triplet', 'var'), RDM_use = RDM_triplet;
else, error('LLM_deepseek: RDM Not Found'); end
r_values_ds = compute_rsm_curve_model(emb_all, wordposition48, subset_indices_41, RDM_use);
dims_ds = 1:length(r_values_ds);
fprintf('  Done. Plateau R = %.4f, dims = %d\n', r_values_ds(end), length(r_values_ds));

%% ===================== 3. MLLM qwen 7B =====================
fprintf('Computing MLLM qwen 7B...\n');
dd = 'data/MLLM_qwen_7B';
emb_all = [load(fullfile(dd, 'qwen_7B_VL_spose_embedding_sorted_final.txt')), ...
            load(fullfile(dd, 'qwen_7B_VL_spose_embedding_sorted_delete.txt'))];
clear RDM_triplet RDM48_triplet;
load(fullfile(dd, 'RDM_48_MLLM_qwen_7B.mat'));
if exist('RDM48_triplet', 'var'), RDM_use = RDM48_triplet;
elseif exist('RDM_triplet', 'var'), RDM_use = RDM_triplet;
else, error('MLLM_qwen_7B: RDM Not Found'); end
r_values_mllm7b = compute_rsm_curve_model(emb_all, wordposition48, subset_indices_41, RDM_use);
dims_mllm7b = 1:length(r_values_mllm7b);
fprintf('  Done. Plateau R = %.4f, dims = %d\n', r_values_mllm7b(end), length(r_values_mllm7b));

%% ===================== 4. MLLM qwen 72B =====================
fprintf('Computing MLLM qwen 72B...\n');
dd = 'data/MLLM_qwen_72B';
emb_all = [load(fullfile(dd, 'qwen_72B_VL_spose_embedding_sorted_final.txt')), ...
            load(fullfile(dd, 'qwen_72B_VL_spose_embedding_sorted_delete.txt'))];
clear RDM_triplet RDM48_triplet;
load(fullfile(dd, 'RDM_48_MLLM_qwen_72B.mat'));
if exist('RDM48_triplet', 'var'), RDM_use = RDM48_triplet;
elseif exist('RDM_triplet', 'var'), RDM_use = RDM_triplet;
else, error('MLLM_qwen_72B: RDM Not Found'); end
r_values_mllm72b = compute_rsm_curve_model(emb_all, wordposition48, subset_indices_41, RDM_use);
dims_mllm72b = 1:length(r_values_mllm72b);
fprintf('  Done. Plateau R = %.4f, dims = %d\n', r_values_mllm72b(end), length(r_values_mllm72b));

%% ===================== 5. Human (3 samples) =====================
fprintf('Computing Human (3 samples)...\n');
dd = 'data/Human';

emb_h1 = [load(fullfile(dd, 'spose_embedding_sorted_human_full_sample_1.txt')), ...
           load(fullfile(dd, 'spose_embedding_sorted_human_full_sample_delete_1.txt'))];
emb_h2 = [load(fullfile(dd, 'spose_embedding_sorted_human_full_sample_2.txt')), ...
           load(fullfile(dd, 'spose_embedding_sorted_human_full_sample_delete_2.txt'))];
emb_h3 = [load(fullfile(dd, 'spose_embedding_sorted_human_full_sample_3.txt')), ...
           load(fullfile(dd, 'spose_embedding_sorted_human_full_sample_delete_3.txt'))];

clear RDM_triplet RDM48_triplet;
load(fullfile(dd, 'RDM_human.mat'));

n_dims_human = size(emb_h1, 2);
dims_human = 1:n_dims_human;
r_values_human = zeros(1, n_dims_human);
emb_samples = {emb_h1, emb_h2, emb_h3};

for d = 1:n_dims_human
    sim_avg = zeros(size(emb_h1, 1));
    
    for s = 1:3
        emb_d = emb_samples{s}(:, 1:d);
        dp_d = emb_d * emb_d';
        esim_d = exp(dp_d);
        n_obj = size(esim_d, 1);
        
        cp = zeros(n_obj, n_obj);
        for i = 1:n_obj
            for j = i+1:n_obj
                ctmp = zeros(1, n_obj);
                for k = 1:n_obj
                    if k == i || k == j, continue; end
                    ctmp(k) = esim_d(i,j) / (esim_d(i,j) + esim_d(i,k) + esim_d(j,k));
                end
                cp(i,j) = sum(ctmp);
            end
        end
        cp = cp / n_obj;
        cp = cp + cp';
        cp(logical(eye(size(cp)))) = 1;
        sim_avg = sim_avg + cp;
    end
    sim_avg = sim_avg / 3;
    
    r_values_human(d) = corr(squareformq(sim_avg), squareformq(1 - RDM_triplet));
end
fprintf('  Done. Plateau R = %.4f, dims = %d\n', r_values_human(end), n_dims_human);

%% ===================== Drawing =====================
customColor1 = [.2 .6 .4];   % LLM_qwen_7B
customColor2 = [.6 .2 .4];   % MLLM_qwen_72B
customColor3 = [.2 .3 .8];   % Human
customColor4 = [.9 .6 .1];   % MLLM_qwen_7B
customColor5 = [.2 .3 .4];   % LLM_deepseek

fig = figure('Position', [500 500 600 500], 'color', 'w');
hold on;

h1 = plot(dims_llm7b,    r_values_llm7b,   '-', 'Color', customColor1, 'LineWidth', 2.5);
h5 = plot(dims_ds,        r_values_ds,       '-', 'Color', customColor5, 'LineWidth', 2.5);
h4 = plot(dims_mllm7b,   r_values_mllm7b,  '-', 'Color', customColor4, 'LineWidth', 2.5);
h2 = plot(dims_mllm72b,  r_values_mllm72b, '-', 'Color', customColor2, 'LineWidth', 2.5);
h3 = plot(dims_human,     r_values_human,    '-', 'Color', customColor3, 'LineWidth', 2.5);

lgd = legend([h3 h1 h5 h4 h2], ...
    {'Human', 'Qwen2.5-7B', 'DeepSeek-R1', 'Qwen2.5-VL-7B', 'Qwen2.5-VL-72B'}, ...
    'Location', 'best', 'FontSize', 14, 'Box', 'off');

ylabel('Representational similarity score', 'FontSize', 18);
xlabel('Number of Dimensions Retained', 'FontSize', 18);
title('Prediction of measured RSM', 'FontSize', 18);

max_dim = max([length(r_values_llm7b), length(r_values_ds), length(r_values_mllm7b), ...
               length(r_values_mllm72b), n_dims_human]);
xlim([1, max_dim + 2]);
ylim([0.15 0.95]);

line([30 30], ylim, 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 1.5, ...
    'HandleVisibility', 'off');

set(gca, 'xtick', [1, 5, 10, 15, 20, 25, 30]);
set(gca, 'Ytick', 0.2:0.1:0.9);

hax = gca;
set(gca, 'FontSize', 18);
hax.Box = 'off';
hax.LineWidth = 1.5;

exportgraphics(fig, 'figures/r_values_against_dims.pdf', 'ContentType', 'vector');


%% ===================== Summary =====================
fprintf('\n============ Plateau R Summary ============\n');
fprintf('  Human:          R = %.4f (dims = %d)\n', r_values_human(end), n_dims_human);
fprintf('  LLM Qwen-7B:    R = %.4f (dims = %d)\n', r_values_llm7b(end), length(r_values_llm7b));
fprintf('  LLM DeepSeek:   R = %.4f (dims = %d)\n', r_values_ds(end), length(r_values_ds));
fprintf('  MLLM Qwen-7B:   R = %.4f (dims = %d)\n', r_values_mllm7b(end), length(r_values_mllm7b));
fprintf('  MLLM Qwen-72B:  R = %.4f (dims = %d)\n', r_values_mllm72b(end), length(r_values_mllm72b));
fprintf('===========================================\n');


function r_values = compute_rsm_curve_model(spose_embedding_all, wordposition48, subset_indices_41, RDM_triplet)
    n_dims = size(spose_embedding_all, 2);
    n_total = size(spose_embedding_all, 1);
    n_wordposition = numel(wordposition48);
    r_values = zeros(1, n_dims);

    assert(n_total == 355, 'Model embeddings must have 355 rows for the original object space.');
    assert(ismember(n_wordposition, [41, 48]), 'wordposition48 must have 41 or 48 entries.');
    assert(max(wordposition48) <= n_total, 'wordposition48 contains indices larger than the embedding row count.');
    assert(all(size(RDM_triplet) >= [max(subset_indices_41) max(subset_indices_41)]), ...
        'Model RDM_triplet is smaller than required by subset_indices_41.');
    if n_wordposition == 48
        assert(max(subset_indices_41) <= n_wordposition, ...
            'subset_indices_41 must index the 48-object space produced by wordposition48.');
    end
    
    for d = 1:n_dims
        spose_embedding_d = spose_embedding_all(:, 1:d);
        dot_product_d = spose_embedding_d * spose_embedding_d';
        esim_d = exp(dot_product_d);
        
        cp = zeros(n_total, n_total);
        for i = 1:n_total
            for j = i+1:n_total
                ctmp = zeros(1, n_total);
                for k_ind = 1:n_wordposition
                    k = wordposition48(k_ind);
                    if k == i || k == j, continue; end
                    ctmp(k) = esim_d(i,j) / (esim_d(i,j) + esim_d(i,k) + esim_d(j,k));
                end
                cp(i,j) = sum(ctmp);
            end
        end
        cp = cp / n_wordposition;
        cp = cp + cp';
        cp(logical(eye(size(cp)))) = 1;

        if n_wordposition == 48
            spose_sim48_d = cp(wordposition48, wordposition48);
            spose_sim_subset_d = spose_sim48_d(subset_indices_41, subset_indices_41);
        else
            spose_sim_subset_d = cp(wordposition48, wordposition48);
        end
        RDM_subset_triplet = RDM_triplet(subset_indices_41, subset_indices_41);
        r_values(d) = corr(squareformq(spose_sim_subset_d), squareformq(1 - RDM_subset_triplet));
    end
end
