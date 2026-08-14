clearvars;
close all;
clc;

% Run this script from any working directory.
base_dir = fileparts(mfilename('fullpath'));
if isempty(base_dir)
    base_dir = pwd;
end

addpath(base_dir);
addpath(genpath(fullfile(base_dir, 'helper_functions')));

%% Run settings
dosave = true;
% 1=Human, 2=Qwen2.5-7B, 3=DeepSeek-R1,
% 4=Qwen2.5-VL-7B, 5=Qwen2.5-VL-72B.
models_to_run = 1:5;  % Use a subset such as [1 5] to run selected models.

index_file_path = fullfile(base_dir, 'data/folder_list', 'folder_list_full_sample_41.txt');
T_indices = readtable(index_file_path, ...
    'FileType', 'text', ...
    'ReadVariableNames', false);
action_indices = T_indices.Var1 + 1;

model_configs = create_model_configs(base_dir);
if any(models_to_run < 1) || any(models_to_run > numel(model_configs))
    error('models_to_run must contain indices between 1 and %d.', numel(model_configs));
end

set(groot, 'DefaultAxesFontName', 'Arial');
set(groot, 'DefaultTextFontName', 'Arial');

%% Compute and draw all selected models
for model_index = models_to_run
    cfg = model_configs(model_index);
    fprintf('\nProcessing %s...\n', cfg.display_name);

    spose_sim41 = load_spose_similarity(cfg.embedding_paths);
    RDM41_triplet = load_rdm41(cfg, action_indices);

    if ~isequal(size(spose_sim41), size(RDM41_triplet))
        error(['Similarity and RDM sizes differ for %s: similarity=%dx%d, ', ...
               'RDM=%dx%d.'], ...
              cfg.display_name, ...
              size(spose_sim41, 1), size(spose_sim41, 2), ...
              size(RDM41_triplet, 1), size(RDM41_triplet, 2));
    end

    [r41, p_value, r41_ci95_lower, r41_ci95_upper] = ...
        bootstrap_rsm_correlation(spose_sim41, RDM41_triplet);

    fprintf('  Pearson r = %.4f, p = %.4g, 95%% CI = [%.4f, %.4f]\n', ...
        r41, p_value, r41_ci95_lower, r41_ci95_upper);

    fig = draw_rsm_triptych( ...
        spose_sim41, RDM41_triplet, ...
        r41, r41_ci95_lower, r41_ci95_upper, cfg);

    if dosave
        output_path = fullfile(base_dir, cfg.output_filename);
        exportgraphics(fig, output_path, ...
            'ContentType', 'vector', ...
            'Resolution', 600);
        fprintf('  Saved: %s\n', output_path);
    end
end

%% =========================== Local Functions ===========================
function configs = create_model_configs(base_dir)
    empty_config = struct( ...
        'display_name', '', ...
        'embedding_paths', {{}}, ...
        'rdm_path', '', ...
        'rdm_variable', '', ...
        'subset_rdm', false, ...
        'scatter_color', [0 0 0], ...
        'output_filename', '', ...
        'show_human_title', false, ...
        'show_72b_rsm_labels', false);

    configs = repmat(empty_config, 5, 1);

    configs(1).display_name = 'Human';
    configs(1).embedding_paths = {
        fullfile(base_dir, 'data', 'Human', ...
            'spose_embedding_sorted_human_full_sample_1.txt')
        fullfile(base_dir, 'data', 'Human', ...
            'spose_embedding_sorted_human_full_sample_2.txt')
        fullfile(base_dir, 'data', 'Human', ...
            'spose_embedding_sorted_human_full_sample_3.txt')
    };
    configs(1).rdm_path = fullfile(base_dir, 'data', 'Human', 'RDM_human.mat');
    configs(1).rdm_variable = 'RDM_triplet';
    configs(1).subset_rdm = false;
    configs(1).scatter_color = [0.2 0.3 0.8];
    configs(1).output_filename = 'RSM_Human_41.pdf';
    configs(1).show_human_title = false;

    configs(2).display_name = 'Qwen2.5-7B';
    configs(2).embedding_paths = {
        fullfile(base_dir, 'data', 'LLM_qwen_7B', ...
            'qwen_7B_spose_embedding_sorted_full_sample.txt')
    };
    configs(2).rdm_path = fullfile(base_dir, 'data', 'LLM_qwen_7B', ...
        'RDM_48_LLM_qwen_7B.mat');
    configs(2).rdm_variable = 'RDM_triplet';
    configs(2).subset_rdm = true;
    configs(2).scatter_color = [0.2 0.6 0.4];
    configs(2).output_filename = 'RSM_Qwen2_5_7B_41.pdf';

    configs(3).display_name = 'DeepSeek-R1';
    configs(3).embedding_paths = {
        fullfile(base_dir, 'data', 'LLM_deepseek', ...
            'deepseek_spose_embedding_sorted_full_sample.txt')
    };
    configs(3).rdm_path = fullfile(base_dir, 'data', 'LLM_deepseek', ...
        'RDM_48_LLM_deepseek.mat');
    configs(3).rdm_variable = 'RDM48_triplet';
    configs(3).subset_rdm = true;
    configs(3).scatter_color = [0.2 0.3 0.4];
    configs(3).output_filename = 'RSM_Deepseek_R1_41.pdf';

    configs(4).display_name = 'Qwen2.5-VL-7B';
    configs(4).embedding_paths = {
        fullfile(base_dir, 'data', 'MLLM_qwen_7B', ...
            'qwen_7B_VL_spose_embedding_sorted_full_sample.txt')
    };
    configs(4).rdm_path = fullfile(base_dir, 'data', 'MLLM_qwen_7B', ...
        'RDM_48_MLLM_qwen_7B.mat');
    configs(4).rdm_variable = 'RDM_triplet';
    configs(4).subset_rdm = true;
    configs(4).scatter_color = [0.9 0.6 0.1];
    configs(4).output_filename = 'RSM_Qwen2_5_VL_7B_41.pdf';

    configs(5).display_name = 'Qwen2.5-VL-72B';
    configs(5).embedding_paths = {
        fullfile(base_dir, 'data', 'MLLM_qwen_72B', ...
            'qwen_72B_spose_embedding_sorted_full_sample.txt')
    };
    configs(5).rdm_path = fullfile(base_dir, 'data', 'MLLM_qwen_72B', ...
        'RDM_48_MLLM_qwen_72B.mat');
    configs(5).rdm_variable = 'RDM_triplet';
    configs(5).subset_rdm = true;
    configs(5).scatter_color = [0.6 0.2 0.4];
    configs(5).output_filename = 'RSM_Qwen2_5_VL_72B_41.pdf';
    configs(5).show_72b_rsm_labels = true;
end

function spose_sim = load_spose_similarity(embedding_paths)
    spose_sum = [];

    for embedding_index = 1:numel(embedding_paths)
        embedding_path = embedding_paths{embedding_index};
        if ~isfile(embedding_path)
            error('Embedding file not found: %s', embedding_path);
        end

        fprintf('  Loading embedding %d/%d: %s\n', ...
            embedding_index, numel(embedding_paths), embedding_path);
        embedding = load(embedding_path);
        current_similarity = compute_choice_probability(embedding);

        if isempty(spose_sum)
            spose_sum = zeros(size(current_similarity));
        elseif ~isequal(size(spose_sum), size(current_similarity))
            error('Human embedding similarity matrices have inconsistent sizes.');
        end

        spose_sum = spose_sum + current_similarity;
    end

    spose_sim = spose_sum / numel(embedding_paths);
end

function spose_sim = compute_choice_probability(spose_embedding)
    dot_product = spose_embedding * spose_embedding';
    esim = exp(dot_product);
    n_objects = size(esim, 1);
    cp = zeros(n_objects, n_objects);

    tic;
    for i = 1:n_objects
        for j = i+1:n_objects
            ctmp = zeros(1, n_objects);
            for k = 1:n_objects
                if k == i || k == j
                    continue;
                end
                ctmp(k) = esim(i,j) / ...
                    (esim(i,j) + esim(i,k) + esim(j,k));
            end
            cp(i,j) = sum(ctmp);
        end
    end
    fprintf('    Choice-probability computation: %.2f s\n', toc);

    cp = cp / n_objects;
    cp = cp + cp';
    cp(logical(eye(size(cp)))) = 1;
    spose_sim = cp;
end

function rdm41 = load_rdm41(cfg, action_indices)
    if ~isfile(cfg.rdm_path)
        error('RDM file not found: %s', cfg.rdm_path);
    end

    rdm_data = load(cfg.rdm_path);
    if ~isfield(rdm_data, cfg.rdm_variable)
        error('Variable %s not found in %s.', cfg.rdm_variable, cfg.rdm_path);
    end
    rdm = rdm_data.(cfg.rdm_variable);

    if cfg.subset_rdm
        if size(rdm, 1) < max(action_indices) || ...
                size(rdm, 2) < max(action_indices)
            error('RDM in %s is smaller than the requested action indices.', ...
                cfg.rdm_path);
        end
        rdm41 = rdm(action_indices, action_indices);
    else
        rdm41 = rdm;
    end
end

function [r_value, p_value, ci_lower, ci_upper] = ...
        bootstrap_rsm_correlation(spose_sim, rdm)
    predicted = squareformq(spose_sim);
    measured = squareformq(1 - rdm);
    [r_value, p_value] = corr(predicted, measured, 'Type', 'Pearson');

    rng(2);
    n_pairs = numel(predicted);
    random_indices = randi(n_pairs, n_pairs, 1000);
    r_boot = zeros(1, 1000);
    for bootstrap_index = 1:1000
        r_boot(bootstrap_index) = corr( ...
            predicted(random_indices(:, bootstrap_index)), ...
            measured(random_indices(:, bootstrap_index)));
    end

    fisher_boot_std = std(atanh(r_boot), [], 2);
    ci_lower = tanh(atanh(r_value) - 1.96 * fisher_boot_std);
    ci_upper = tanh(atanh(r_value) + 1.96 * fisher_boot_std);
end

function fig = draw_rsm_triptych( ...
        spose_sim, rdm, r_value, ci_lower, ci_upper, cfg)
    disp('  Creating publication-quality figure...');

    figure_width = 1800;
    figure_height = 600;
    fig = figure( ...
        'Position', [100 100 figure_width figure_height], ...
        'Color', 'white', ...
        'Units', 'pixels', ...
        'Name', cfg.display_name);

    measured_rsm = 1 - rdm;
    n_objects = size(spose_sim, 1);
    fit_line_color = [220 50 32] / 255;

    % Shared typography for all five figures. For elements that differed
    % across the original scripts, use the larger original font size.
    font_sizes = struct( ...
        'axes', 26, ...
        'action_count', 24, ...
        'rsm_panel_label', 28, ...
        'colorbar_ticks', 24, ...
        'colorbar_label', 24, ...
        'axis_label', 24, ...
        'statistics', 18, ...
        'panel_title', 24, ...
        'human_title', 28);

    %% Panel 1: predicted RSM
    ax1 = subplot(1, 3, 1, 'Parent', fig);
    imagesc(ax1, spose_sim);
    colormap(ax1, slanCM(102));
    caxis(ax1, [0 1]);
    axis(ax1, 'equal');
    axis(ax1, 'tight');
    box(ax1, 'on');
    set(ax1, ...
        'FontSize', font_sizes.axes, ...
        'LineWidth', 1.5, ...
        'TickDir', 'out', ...
        'XColor', 'k', ...
        'YColor', 'k');

    if cfg.show_72b_rsm_labels
        text(ax1, 0.5, -0.2, 'Predicted RSM', ...
            'Units', 'normalized', ...
            'HorizontalAlignment', 'center', ...
            'FontSize', font_sizes.rsm_panel_label, ...
            'FontWeight', 'bold');
    end

    text(ax1, 0.5, 1.05, sprintf('%d diverse actions', n_objects), ...
        'Units', 'normalized', ...
        'HorizontalAlignment', 'center', ...
        'FontSize', font_sizes.action_count, ...
        'FontWeight', 'bold');

    %% Panel 2: measured RSM
    ax2 = subplot(1, 3, 2, 'Parent', fig);
    imagesc(ax2, measured_rsm);
    colormap(ax2, slanCM(102));
    caxis(ax2, [0 1]);
    axis(ax2, 'equal');
    axis(ax2, 'tight');
    box(ax2, 'on');
    set(ax2, ...
        'FontSize', font_sizes.axes, ...
        'LineWidth', 1.5, ...
        'TickDir', 'out', ...
        'XColor', 'k', ...
        'YColor', 'k');

    if cfg.show_72b_rsm_labels
        text(ax2, 0.5, -0.2, 'Measured RSM', ...
            'Units', 'normalized', ...
            'HorizontalAlignment', 'center', ...
            'FontSize', font_sizes.rsm_panel_label, ...
            'FontWeight', 'bold');
    end

    cb = colorbar(ax2, 'Location', 'eastoutside');
    cb.Position = [0.038 0.28 0.01 0.54];
    cb.FontSize = font_sizes.colorbar_ticks;
    cb.Label.String = 'Similarity';
    cb.Label.FontSize = font_sizes.colorbar_label;
    cb.Label.FontWeight = 'bold';
    cb.Label.Rotation = 90;
    cb.Label.VerticalAlignment = 'bottom';

    %% Panel 3: RSM correlation
    ax3 = subplot(1, 3, 3, 'Parent', fig);
    x_data = squareformq(spose_sim);
    y_data = squareformq(measured_rsm);

    scatter(ax3, x_data, y_data, 35, ...
        'MarkerFaceColor', cfg.scatter_color, ...
        'MarkerFaceAlpha', 0.6, ...
        'MarkerEdgeColor', 'none');
    hold(ax3, 'on');

    fit_coefficients = polyfit(x_data, y_data, 1);
    fit_x = [0 1];
    fit_y = polyval(fit_coefficients, fit_x);
    plot(ax3, fit_x, fit_y, ...
        'Color', fit_line_color, ...
        'LineWidth', 2.5, ...
        'LineStyle', '--');
    plot(ax3, [0 1], [0 1], ...
        'Color', [0.3 0.3 0.3], ...
        'LineWidth', 1.5, ...
        'LineStyle', ':');

    xlim(ax3, [0 1]);
    ylim(ax3, [0 1]);
    axis(ax3, 'square');
    box(ax3, 'on');
    grid(ax3, 'on');
    grid(ax3, 'minor');
    set(ax3, ...
        'FontSize', font_sizes.axes, ...
        'LineWidth', 1.5, ...
        'TickDir', 'out', ...
        'GridAlpha', 0.2, ...
        'MinorGridAlpha', 0.1);

    xlabel(ax3, 'Predicted similarity', ...
        'FontSize', font_sizes.axis_label, ...
        'FontWeight', 'bold');
    ylabel(ax3, 'Measured similarity', ...
        'FontSize', font_sizes.axis_label, ...
        'FontWeight', 'bold');

    stats_text = sprintf( ...
        ['r = %.2f [%.2f, %.2f]\\newline', ...
         'P < 0.001\\newline', ...
         '\\color{red}-- \\color[rgb]{0.3,0.3,0.3}Linear fit'], ...
        r_value, ci_lower, ci_upper);
    title(ax3, 'Model Performance', ...
        'FontSize', font_sizes.panel_title, ...
        'FontWeight', 'bold');

    %% Match the layout used by the five original scripts
    new_width = 0.22;
    new_height = 0.7;
    set(ax1, 'Position', [0.082 0.2 new_width new_height]);
    set(ax2, 'Position', [0.343 0.2 new_width new_height]);
    set(ax3, 'Position', [0.63 0.2 new_width new_height]);

    %% Semi-transparent statistics box
    % annotation textboxes support FaceAlpha, whereas axes text objects do
    % not provide a background-alpha property. Position is normalized to
    % the figure and derived from the final ax3 position.
    stats_box_alpha = 0.55;  % 0 = transparent; 1 = opaque
    ax3_position = ax3.Position;
    stats_box_position = [ ...
        ax3_position(1) + 0.02 * ax3_position(3), ...
        ax3_position(2) + 0.67 * ax3_position(4), ...
        0.64 * ax3_position(3), ...
        0.25 * ax3_position(4)];

    annotation(fig, 'textbox', stats_box_position, ...
        'String', stats_text, ...
        'FontSize', font_sizes.statistics, ...
        'FontWeight', 'bold', ...
        'BackgroundColor', [1 1 1], ...
        'FaceAlpha', stats_box_alpha, ...
        'EdgeColor', 'k', ...
        'LineWidth', 1, ...
        'Margin', 4, ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'top', ...
        'Interpreter', 'tex', ...
        'FitBoxToText', 'off');

    % Preserve the active overall title from the Human script only.
    if cfg.show_human_title
        annotation(fig, 'textbox', [0 0.9 1 0.1], ...
            'String', 'Model Comparison of Predicted and Measured RSMs', ...
            'EdgeColor', 'none', ...
            'HorizontalAlignment', 'center', ...
            'FontSize', font_sizes.human_title, ...
            'FontWeight', 'bold');
    end
end

