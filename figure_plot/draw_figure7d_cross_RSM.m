clearvars;
close all;
clc;

base_dir = pwd;
addpath(base_dir);
addpath(genpath(fullfile(base_dir, 'helper_functions')));

%% Shared inputs
index_file_path = fullfile(base_dir, 'data', 'folder_list/folder_list_full_sample_41.txt');
T_indices = readtable(index_file_path, 'FileType', 'text', 'ReadVariableNames', false);
action_indices = T_indices.Var1 + 1;

%% Model setup
nM = 5;
model_names = {'Human', 'Qwen2.5-7B', 'DeepSeek-R1', 'Qwen2.5-VL-7B', 'Qwen2.5-VL-72B'};

% Accent colors only used for subtle diagonal highlight consistency.
model_colors = [
    0.16, 0.33, 0.67;  % Human
    0.19, 0.58, 0.24;  % Qwen2.5-7B
    0.46, 0.34, 0.69;  % DeepSeek-R1
    0.87, 0.52, 0.13;  % Qwen2.5-VL-7B
    0.73, 0.23, 0.33   % Qwen2.5-VL-72B
];

predicted_vec = cell(nM, 1);
measured_vec  = cell(nM, 1);

%% 1) Human
fprintf('Computing Human...\n');
spose_emb_h1 = load(fullfile(base_dir, 'data', 'Human', 'spose_embedding_sorted_human_full_sample_1.txt'));
spose_emb_h2 = load(fullfile(base_dir, 'data', 'Human', 'spose_embedding_sorted_human_full_sample_2.txt'));
spose_emb_h3 = load(fullfile(base_dir, 'data', 'Human', 'spose_embedding_sorted_human_full_sample_3.txt'));

fprintf('  Sample 1...\n'); sim_h1 = compute_cp(spose_emb_h1);
fprintf('  Sample 2...\n'); sim_h2 = compute_cp(spose_emb_h2);
fprintf('  Sample 3...\n'); sim_h3 = compute_cp(spose_emb_h3);
spose_sim_human = (sim_h1 + sim_h2 + sim_h3) / 3;

tmp_human = load(fullfile(base_dir, 'data', 'Human', 'RDM_human.mat'));
if ~isfield(tmp_human, 'RDM_triplet')
    error('Human RDM file does not contain variable RDM_triplet.');
end
RDM41_human = tmp_human.RDM_triplet;

predicted_vec{1} = squareformq(spose_sim_human);
measured_vec{1}  = squareformq(1 - RDM41_human);
fprintf('  Done.\n');

%% 2) Qwen2.5-7B
fprintf('Computing Qwen2.5-7B...\n');
emb = load(fullfile(base_dir, 'data', 'LLM_qwen_7B', 'qwen_7B_spose_embedding_sorted_full_sample.txt'));
fprintf('  Embedding: %d x %d\n', size(emb,1), size(emb,2));
spose_sim = compute_cp(emb);
rdm = load_rdm_subset(fullfile(base_dir, 'data', 'LLM_qwen_7B', 'RDM_48_LLM_qwen_7B.mat'), action_indices);
predicted_vec{2} = squareformq(spose_sim);
measured_vec{2}  = squareformq(1 - rdm);
fprintf('  Done. predicted=%d, measured=%d\n', numel(predicted_vec{2}), numel(measured_vec{2}));

%% 3) DeepSeek-R1
fprintf('Computing DeepSeek-R1...\n');
emb = load(fullfile(base_dir, 'data', 'LLM_deepseek', 'deepseek_spose_embedding_sorted_full_sample.txt'));
fprintf('  Embedding: %d x %d\n', size(emb,1), size(emb,2));
spose_sim = compute_cp(emb);
rdm = load_rdm_subset(fullfile(base_dir, 'data', 'LLM_deepseek', 'RDM_48_LLM_deepseek.mat'), action_indices);
predicted_vec{3} = squareformq(spose_sim);
measured_vec{3}  = squareformq(1 - rdm);
fprintf('  Done. predicted=%d, measured=%d\n', numel(predicted_vec{3}), numel(measured_vec{3}));

%% 4) Qwen2.5-VL-7B
fprintf('Computing Qwen2.5-VL-7B...\n');
emb = load(fullfile(base_dir, 'data', 'MLLM_qwen_7B', 'qwen_7B_VL_spose_embedding_sorted_full_sample.txt'));
fprintf('  Embedding: %d x %d\n', size(emb,1), size(emb,2));
spose_sim = compute_cp(emb);
rdm = load_rdm_subset(fullfile(base_dir, 'data', 'MLLM_qwen_7B', 'RDM_48_MLLM_qwen_7B.mat'), action_indices);
predicted_vec{4} = squareformq(spose_sim);
measured_vec{4}  = squareformq(1 - rdm);
fprintf('  Done. predicted=%d, measured=%d\n', numel(predicted_vec{4}), numel(measured_vec{4}));

%% 5) Qwen2.5-VL-72B
fprintf('Computing Qwen2.5-VL-72B...\n');
emb = load(fullfile(base_dir, 'data', 'MLLM_qwen_72B', 'qwen_72B_spose_embedding_sorted_full_sample.txt'));
fprintf('  Embedding: %d x %d\n', size(emb,1), size(emb,2));
spose_sim = compute_cp(emb);
rdm = load_rdm_subset(fullfile(base_dir, 'data', 'MLLM_qwen_72B', 'RDM_48_MLLM_qwen_72B.mat'), action_indices);
predicted_vec{5} = squareformq(spose_sim);
measured_vec{5}  = squareformq(1 - rdm);
fprintf('  Done. predicted=%d, measured=%d\n', numel(predicted_vec{5}), numel(measured_vec{5}));

%% Vector length checks
fprintf('\n--- Vector size verification ---\n');
all_lengths = zeros(nM, 2);
for m = 1:nM
    all_lengths(m, 1) = numel(predicted_vec{m});
    all_lengths(m, 2) = numel(measured_vec{m});
    fprintf('  %s: predicted=%d, measured=%d\n', model_names{m}, all_lengths(m,1), all_lengths(m,2));
end

unique_lens = unique(all_lengths(:));
if numel(unique_lens) > 1
    error(['Vector lengths are inconsistent. Please check embedding row count ', ...
           'and RDM size before computing 5x5 cross-correlation.']);
end
fprintf('  All vectors: %d elements. Cross-correlation OK.\n', unique_lens);

%% 5x5 cross-RSM correlations
fprintf('\nComputing 5x5 cross-RSM correlation matrix...\n');
R_cross = zeros(nM, nM);
R_cross_p = zeros(nM, nM);

for i = 1:nM
    for j = 1:nM
        [R_cross(i,j), R_cross_p(i,j)] = corr(predicted_vec{i}, measured_vec{j}, 'Type', 'Pearson');
    end
end

fprintf('\n========== Cross-RSM Correlation Matrix ==========' );
fprintf('\n%18s', '');
for j = 1:nM
    fprintf('%14s', model_names{j});
end
fprintf('\n%18s', '');
for j = 1:nM
    fprintf('%14s', '(measured)');
end
fprintf('\n----------------------------------------------------------------------\n');
for i = 1:nM
    fprintf('%-18s', [model_names{i}, ' (pred)']);
    for j = 1:nM
        if i == j
            fprintf('%14s', sprintf('[%.3f]', R_cross(i,j)));
        else
            fprintf('%14.3f', R_cross(i,j));
        end
    end
    fprintf('\n');
end
fprintf('======================================================================\n');

%% Bootstrap 95% CI for diagonal
fprintf('\nBootstrap CI for diagonal (self-fit)...\n');
R_diag_ci = zeros(nM, 2);
rng(2);
for m = 1:nM
    c1 = predicted_vec{m};
    c2 = measured_vec{m};
    n_pairs = numel(c1);

    rnd = randi(n_pairs, n_pairs, 1000);
    r_boot = zeros(1, 1000);
    for b = 1:1000
        r_boot(b) = corr(c1(rnd(:,b)), c2(rnd(:,b)));
    end

    r_val = R_cross(m,m);
    R_diag_ci(m,1) = tanh(atanh(r_val) - 1.96 * std(atanh(r_boot)));
    R_diag_ci(m,2) = tanh(atanh(r_val) + 1.96 * std(atanh(r_boot)));
    fprintf('  %s: R=%.3f [%.3f, %.3f]\n', model_names{m}, r_val, R_diag_ci(m,1), R_diag_ci(m,2));
end

%% Publication-style heatmap
fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 18 16]);
ax = axes('Parent', fig, 'Position', [0.17 0.17 0.64 0.66]);
hold(ax, 'on');

imagesc(ax, R_cross);
axis(ax, 'square');
axis(ax, [0.5 nM+0.5 0.5 nM+0.5]);
set(ax, 'YDir', 'normal');

cmap_custom = make_nature_blue_map(256);
colormap(ax, cmap_custom);

cmin = max(0, min(R_cross(:)) - 0.02);
cmax = min(1, max(R_cross(:)) + 0.02);
if (cmax - cmin) < 0.10
    cpad = 0.05;
    cmin = max(0, cmin - cpad);
    cmax = min(1, cmax + cpad);
end
caxis(ax, [cmin cmax]);

for k = 0.5:1:(nM + 0.5)
    plot(ax, [0.5, nM + 0.5], [k, k], '-', 'Color', [1, 1, 1] * 0.94, 'LineWidth', 0.9);
    plot(ax, [k, k], [0.5, nM + 0.5], '-', 'Color', [1, 1, 1] * 0.94, 'LineWidth', 0.9);
end

clim = caxis(ax);
current_cmap = colormap(ax);
for i = 1:nM
    for j = 1:nM
        if i == j
            rectangle('Position', [j-0.5, i-0.5, 1, 1], ...
                'EdgeColor', model_colors(i,:), 'LineWidth', 2.0, 'Parent', ax);
        end

        cell_rgb = map_value_to_rgb(R_cross(i,j), clim, current_cmap);
        txt_col = choose_text_color(cell_rgb);

        text(ax, j, i, sprintf('%.3f', R_cross(i,j)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontName', 'Helvetica', ...
            'FontSize', 11.5, ...
            'FontWeight', 'bold', ...
            'Color', txt_col);
    end
end

set(ax, 'XTick', 1:nM, ...
        'YTick', 1:nM, ...
        'XTickLabel', model_names, ...
        'YTickLabel', model_names, ...
        'TickDir', 'out', ...
        'TickLength', [0 0], ...
        'LineWidth', 0.9, ...
        'Box', 'on', ...
        'FontName', 'Helvetica', ...
        'FontSize', 10, ...
        'XColor', [0.2 0.2 0.2], ...
        'YColor', [0.2 0.2 0.2]);
xtickangle(ax, 30);

xlabel(ax, 'Measured RSM', 'FontName', 'Helvetica', 'FontSize', 11.5, 'FontWeight', 'bold');
ylabel(ax, 'Predicted RSM', 'FontName', 'Helvetica', 'FontSize', 11.5, 'FontWeight', 'bold');
title(ax, 'Cross-Model RSM Correlation', 'FontName', 'Helvetica', 'FontSize', 13, 'FontWeight', 'bold');

cb = colorbar(ax, 'eastoutside');
cb.Position = [0.83 0.21 0.02 0.58];
cb.Ticks = linspace(cmin, cmax, 5);
cb.Color = [0.2 0.2 0.2];
cb.FontName = 'Helvetica';
cb.FontSize = 9.5;
cb.Label.String = 'Pearson r';
cb.Label.FontName = 'Helvetica';
cb.Label.FontSize = 10.5;

annotation(fig, 'textbox', [0.17 0.07 0.64 0.06], ...
    'EdgeColor', 'none', ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'middle', ...
    'Color', [0.25 0.25 0.25], ...
    'FontName', 'Helvetica', ...
    'FontSize', 9.5);

exportgraphics(fig, fullfile(base_dir, 'cross_RSM_5x5_heatmap.pdf'), 'ContentType', 'vector');

fprintf('\nSaved: cross_RSM_5x5_heatmap_clean.pdf and cross_RSM_5x5_heatmap_clean.png\n');

%% =========================== Local Functions ===========================
function spose_sim = compute_cp(spose_embedding)
    dot_product = spose_embedding * spose_embedding';
    esim = exp(dot_product);
    n_objects = size(esim, 1);
    cp = zeros(n_objects, n_objects);

    for i = 1:n_objects
        for j = i+1:n_objects
            ctmp = zeros(1, n_objects);
            for k = 1:n_objects
                if k == i || k == j
                    continue;
                end
                ctmp(k) = esim(i,j) / (esim(i,j) + esim(i,k) + esim(j,k));
            end
            cp(i,j) = sum(ctmp);
        end
    end

    cp = cp / n_objects;
    cp = cp + cp';
    cp(logical(eye(size(cp)))) = 1;
    spose_sim = cp;
end

function rdm41 = load_rdm_subset(mat_path, action_indices)
    tmp = load(mat_path);

    if isfield(tmp, 'RDM48_triplet')
        rdm = tmp.RDM48_triplet;
    elseif isfield(tmp, 'RDM_triplet')
        rdm = tmp.RDM_triplet;
    else
        error('RDM variable not found in file: %s', mat_path);
    end

    if size(rdm, 1) >= max(action_indices)
        rdm41 = rdm(action_indices, action_indices);
    else
        error('RDM size in %s is smaller than required indices.', mat_path);
    end
end

function cmap = make_nature_blue_map(n)
    anchors = [
        0.965, 0.975, 0.992
        0.855, 0.913, 0.958
        0.706, 0.826, 0.911
        0.493, 0.708, 0.844
        0.258, 0.545, 0.756
        0.101, 0.357, 0.588
    ];

    x = linspace(0, 1, size(anchors, 1));
    xi = linspace(0, 1, n);

    cmap = zeros(n, 3);
    for k = 1:3
        cmap(:, k) = interp1(x, anchors(:, k), xi, 'linear');
    end
end

function rgb = map_value_to_rgb(val, clim, cmap)
    if clim(2) == clim(1)
        idx = 1;
    else
        val_n = (val - clim(1)) / (clim(2) - clim(1));
        val_n = max(0, min(1, val_n));
        idx = round(val_n * (size(cmap, 1) - 1)) + 1;
    end
    rgb = cmap(idx, :);
end

function txt_col = choose_text_color(bg_rgb)
    lum = 0.299 * bg_rgb(1) + 0.587 * bg_rgb(2) + 0.114 * bg_rgb(3);
    if lum > 0.58
        txt_col = [0.10, 0.10, 0.10];
    else
        txt_col = [1, 1, 1];
    end
end
