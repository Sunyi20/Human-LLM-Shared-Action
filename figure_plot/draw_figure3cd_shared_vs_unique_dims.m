clc; clear; close all;

base_dir = pwd;
addpath(base_dir)
addpath(genpath(fullfile(base_dir, 'helper_functions')))

%% Data loading
spose_embedding_human = load(fullfile('data/Human/human_odd_one_out_spose_embedding_sorted_final.txt'));
rows_to_remove = [247, 248, 249];
spose_embedding_human(rows_to_remove, :) = [];

spose_LLM_qwen7B_raw   = load(fullfile('data/LLM_qwen_7B/LLM_qwen_7B_overlap_spose_embedding.txt'));
spose_MLLM_qwen7B_raw  = load(fullfile('data/MLLM_qwen_7B/MLLM_qwen_7B_overlap_spose_embedding.txt'));
spose_LLM_deepseek_raw = load(fullfile('data/LLM_deepseek/LLM_deepseek_overlap_spose_embedding.txt'));
spose_MLLM_qwen72B_raw = load(fullfile('data/MLLM_qwen_72B/MLLM_qwen_72B_overlap_spose_embedding.txt'));

%% Data alignment
n_repeats = 3;
nH = size(spose_embedding_human, 1);
if mod(nH, n_repeats) ~= 0
    error('Human embedding row count must be divisible by n_repeats.');
end
expected_rows = nH / n_repeats;

assert(size(spose_LLM_qwen7B_raw, 1) == expected_rows, 'Qwen2.5-7B row count mismatch.');
assert(size(spose_MLLM_qwen7B_raw, 1) == expected_rows, 'Qwen2.5-VL-7B row count mismatch.');
assert(size(spose_LLM_deepseek_raw, 1) == expected_rows, 'Deepseek-R1 row count mismatch.');
assert(size(spose_MLLM_qwen72B_raw, 1) == expected_rows, 'Qwen2.5-VL-72B row count mismatch.');

spose_LLM_qwen7B   = repelem(spose_LLM_qwen7B_raw, n_repeats, 1);
spose_MLLM_qwen7B  = repelem(spose_MLLM_qwen7B_raw, n_repeats, 1);
spose_LLM_deepseek = repelem(spose_LLM_deepseek_raw, n_repeats, 1);
spose_MLLM_qwen72B = repelem(spose_MLLM_qwen72B_raw, n_repeats, 1);

%% Model groups
group_data = {
    spose_embedding_human, spose_LLM_qwen7B,   spose_MLLM_qwen7B;
    spose_embedding_human, spose_LLM_qwen7B,   spose_LLM_deepseek;
    spose_embedding_human, spose_MLLM_qwen7B,  spose_MLLM_qwen72B;
    spose_embedding_human, spose_LLM_deepseek, spose_MLLM_qwen72B;
};

group_names = {
    {'Human', 'Qwen2.5-7B', 'Qwen2.5-VL-7B'};
    {'Human', 'Qwen2.5-7B', 'Deepseek-R1'};
    {'Human', 'Qwen2.5-VL-7B', 'Qwen2.5-VL-72B'};
    {'Human', 'Deepseek-R1', 'Qwen2.5-VL-72B'};
};

group_filenames = {
    'venn3_Human_LLMqwen7B_MLLMqwen7B';
    'venn3_Human_LLMqwen7B_LLMdeepseek';
    'venn3_Human_MLLMqwen7B_MLLMqwen72B';
    'venn3_Human_LLMdeepseek_MLLMqwen72B';
};

all_model_names = {'Human', 'Qwen2.5-7B', 'Qwen2.5-VL-7B', 'Deepseek-R1', 'Qwen2.5-VL-72B'};
% Keep the original model-to-color identity used in earlier figures, but
% slightly mute the palette for a cleaner journal-style presentation.
all_model_colors = [
    0.22 0.34 0.76  % Human: blue
    0.24 0.58 0.42  % Qwen2.5-7B: green
    0.82 0.60 0.18  % Qwen2.5-VL-7B: amber
    0.28 0.34 0.42  % Deepseek-R1: slate
    0.56 0.30 0.47  % Qwen2.5-VL-72B: plum
];

threshold = 0.2;
nGroups = size(group_data, 1);

%% Group-wise analysis and plotting
for g = 1:nGroups
    setName = group_names{g};
    fprintf('\n========== Group %d: %s / %s / %s ==========\n', g, setName{1}, setName{2}, setName{3});

    A = group_data{g, 1};
    B = group_data{g, 2};
    C = group_data{g, 3};

    [nA, dim_A] = size(A);
    [nB, dim_B] = size(B);
    [nC, dim_C] = size(C);

    fprintf('A (%s): %d x %d\n', setName{1}, nA, dim_A);
    fprintf('B (%s): %d x %d\n', setName{2}, nB, dim_B);
    fprintf('C (%s): %d x %d\n', setName{3}, nC, dim_C);

    c_adapted_AB = compute_adapted_corr(A, B);
    c_adapted_AC = compute_adapted_corr(A, C);
    c_adapted_BC = compute_adapted_corr(B, C);

    result_A = false(dim_A, 3);
    result_B = false(dim_B, 3);
    result_C = false(dim_C, 3);

    dims_A = 1:dim_A;
    dims_B = 1:dim_B;
    dims_C = 1:dim_C;

    result_A(:, 1) = true;
    result_B(:, 2) = true;
    result_C(:, 3) = true;

    for i = dims_A
        for j = 1:dim_B
            if ismember(j, dims_B) && c_adapted_AB(i, j) >= threshold
                result_A(i, 2) = true;
                result_B(j, 2) = false;
                dims_B(dims_B == j) = [];
                break;
            end
        end

        for j = 1:dim_C
            if ismember(j, dims_C) && c_adapted_AC(i, j) >= threshold
                result_A(i, 3) = true;
                result_C(j, 3) = false;
                dims_C(dims_C == j) = [];
                break;
            end
        end
    end

    for i = dims_B
        for j = 1:dim_C
            if ismember(j, dims_C) && c_adapted_BC(i, j) >= threshold
                result_B(i, 3) = true;
                result_C(j, 3) = false;
                dims_C(dims_C == j) = [];
                break;
            end
        end
    end

    result = [result_A; result_B; result_C];
    dim_origin = [
        ones(dim_A, 1), (1:dim_A)';
        2 * ones(dim_B, 1), (1:dim_B)';
        3 * ones(dim_C, 1), (1:dim_C)'
    ];

    valid_rows = any(result, 2);
    result = result(valid_rows, :);
    dim_origin = dim_origin(valid_rows, :);

    shared_ABC = sum(all(result, 2));
    shared_AB  = sum(result(:, 1) & result(:, 2) & ~result(:, 3));
    shared_AC  = sum(result(:, 1) & ~result(:, 2) & result(:, 3));
    shared_BC  = sum(~result(:, 1) & result(:, 2) & result(:, 3));
    unique_A   = sum(result(:, 1) & ~result(:, 2) & ~result(:, 3));
    unique_B   = sum(~result(:, 1) & result(:, 2) & ~result(:, 3));
    unique_C   = sum(~result(:, 1) & ~result(:, 2) & result(:, 3));

    fprintf('\n--- Summary ---\n');
    fprintf('Shared (A,B,C): %d\n', shared_ABC);
    fprintf('Shared (A,B):   %d\n', shared_AB);
    fprintf('Shared (A,C):   %d\n', shared_AC);
    fprintf('Shared (B,C):   %d\n', shared_BC);
    fprintf('Unique A:       %d\n', unique_A);
    fprintf('Unique B:       %d\n', unique_B);
    fprintf('Unique C:       %d\n', unique_C);
    fprintf('Total rows:     %d\n', size(result, 1));

    regions = {
        'unique_A',   result(:,1) & ~result(:,2) & ~result(:,3), setName{1};
        'unique_B',  ~result(:,1) &  result(:,2) & ~result(:,3), setName{2};
        'unique_C',  ~result(:,1) & ~result(:,2) &  result(:,3), setName{3};
        'shared_AB',  result(:,1) &  result(:,2) & ~result(:,3), [setName{1} ' & ' setName{2}];
        'shared_AC',  result(:,1) & ~result(:,2) &  result(:,3), [setName{1} ' & ' setName{3}];
        'shared_BC', ~result(:,1) &  result(:,2) &  result(:,3), [setName{2} ' & ' setName{3}];
        'shared_ABC', result(:,1) &  result(:,2) &  result(:,3), [setName{1} ' & ' setName{2} ' & ' setName{3}];
    };

    fprintf('\n--- Region Details ---\n');
    for r = 1:size(regions, 1)
        region_name = regions{r, 1};
        region_mask = regions{r, 2};
        region_desc = regions{r, 3};
        region_count = sum(region_mask);

        if region_count == 0
            fprintf('  [%s] (%s): 0 dimensions\n', region_name, region_desc);
            continue;
        end

        fprintf('  [%s] (%s): %d dimensions\n', region_name, region_desc, region_count);
        idx = find(region_mask);
        for src = 1:3
            src_rows = idx(dim_origin(idx, 1) == src);
            if isempty(src_rows)
                continue;
            end
            dim_indices = dim_origin(src_rows, 2)';
            fprintf('    From %s (%d): [%s]\n', setName{src}, numel(dim_indices), num2str(dim_indices));
        end
    end

    fprintf('\n--- Unique Dimensions ---\n');
    for src = 1:3
        if src == 1
            mask = result(:,1) & ~result(:,2) & ~result(:,3);
        elseif src == 2
            mask = ~result(:,1) & result(:,2) & ~result(:,3);
        else
            mask = ~result(:,1) & ~result(:,2) & result(:,3);
        end

        idx = find(mask);
        src_dims = dim_origin(idx(dim_origin(idx, 1) == src), 2)';
        if isempty(src_dims)
            fprintf('  %s: none\n', setName{src});
        else
            fprintf('  %s (%d): [%s]\n', setName{src}, numel(src_dims), num2str(src_dims));
        end
    end

    currentColors = get_group_colors(setName, all_model_names, all_model_colors);
    region_counts = [unique_A, unique_B, unique_C, shared_AB, shared_AC, shared_BC, shared_ABC];

    create_publication_venn(setName, region_counts, currentColors, ['figures/' group_filenames{g}]);
    create_publication_upset(result, setName, currentColors, ['figures/' group_filenames{g} '_upset']);
end

%% Separate legend
create_model_legend(all_model_names, all_model_colors, 'figures/model_colors_legend');



fprintf('\nAll %d groups finished.\n', nGroups);

%% Local functions
function c_adapted = compute_adapted_corr(X, Y)
    dim_X = size(X, 2);
    dim_Y = size(Y, 2);

    c = corr(X, Y);
    [~, ii] = max(c);
    for i = 2:dim_Y
        if any(ii(1:i-1) == ii(i))
            ii(i) = 1000;
        end
    end
    [~, si] = sort(ii);

    c_base = corr(X) - eye(dim_X);
    if dim_Y > dim_X
        c_base_adj = [c_base, zeros(dim_X, dim_Y - dim_X)];
    elseif dim_Y < dim_X
        c_base_adj = c_base(:, 1:dim_Y);
    else
        c_base_adj = c_base;
    end

    c_adapted = c(:, si) - c_base_adj;
end

function currentColors = get_group_colors(setName, all_model_names, all_model_colors)
    currentColors = zeros(3, 3);
    for k = 1:3
        match_idx = find(strcmp(all_model_names, setName{k}), 1);
        currentColors(k, :) = all_model_colors(match_idx, :);
    end
end

function create_publication_venn(setName, region_counts, currentColors, out_base)
    % The local venn.m draws circles in the order: left, right, top.
    % But setName is interpreted as: top, left, right.
    % Reorder colors so fill colors match the displayed set labels.
    vennColors = lighten_colors(currentColors([2, 3, 1], :), 0.18);
    venn_fig = venn(3, 'sets', setName, 'labels', string(region_counts), ...
        'alpha', 1.0, ...
        'edgeC', [1 1 1], ...
        'colors', vennColors, ...
        'edgeW', 2.2, ...
        'labelC', [0.15 0.15 0.15]);

    set(venn_fig, 'Color', 'w', 'Units', 'centimeters', 'Position', [2 2 13 10]);
    ax = findobj(venn_fig, 'Type', 'axes');
    if ~isempty(ax)
        ax = ax(1);
        axis(ax, 'equal');
        axis(ax, 'off');
%         title(ax, sprintf('%s  |  %s  |  %s', setName{1}, setName{2}, setName{3}), ...
%             'FontName', 'Helvetica', 'FontSize', 12, 'FontWeight', 'bold', ...
%             'Color', [0.12 0.12 0.12]);
    end

    txt = findall(venn_fig, 'Type', 'text');
    for i = 1:numel(txt)
        current_str = string(txt(i).String);
        set(txt(i), 'FontName', 'Helvetica');
        if any(strcmp(current_str, string(setName)))
            idx = find(strcmp(current_str, string(setName)), 1);
            set(txt(i), 'FontSize', 9.5, 'FontWeight', 'bold', 'Color', lighten_colors(currentColors(idx, :), 0.10));
        else
            set(txt(i), 'FontSize', 20, 'FontWeight', 'bold', 'Color', [0.14 0.14 0.14]);
        end
    end

    exportgraphics(venn_fig, [out_base '.pdf'], 'ContentType', 'vector');
    close(venn_fig);
end

function create_publication_upset(Data, setName, currentColors, out_base)
    pBool = abs(dec2bin((1:(2^size(Data,2)-1))')) - 48;
    [pPos, ~] = find(((pBool * (1 - Data')) | ((1 - pBool) * Data')) == 0);
    sPPos = sort(pPos);
    dPPos = find([diff(sPPos); 1]);
    pType = sPPos(dPPos);
    pCount = diff([0; dPPos]);
    [pCount, pInd] = sort(pCount, 'descend');
    pType = pType(pInd);

    sCount = sum(Data, 1);
    [~, sInd] = sort(sCount, 'descend');
    sType = 1:size(Data, 2);
    sType = sType(sInd);

    lineColor = [0.20 0.22 0.25];
    dotColor = [0.87 0.88 0.90];
    stripeColor = [0.965 0.968 0.972; 0.985 0.987 0.990];

    fig = figure('Units', 'centimeters', 'Position', [2 2 18 12], 'Color', 'w');

    axI = axes('Parent', fig);
    hold(axI, 'on');
    set(axI, 'Position', [0.16 0.39 0.78 0.48], ...
        'LineWidth', 0.9, ...
        'Box', 'off', ...
        'TickDir', 'out', ...
        'FontName', 'Helvetica', ...
        'FontSize', 14.5, ...
        'XTick', [], ...
        'XLim', [0, length(pType) + 1], ...
        'XColor', [0.18 0.18 0.18], ...
        'YColor', [0.18 0.18 0.18]);
    ylabel(axI, 'Intersection Size', 'FontName', 'Helvetica', 'FontSize', 15.5, 'FontWeight', 'bold');
%     title(axI, sprintf('Shared and unique dimensions: %s | %s | %s', setName{1}, setName{2}, setName{3}), ...
%         'FontName', 'Helvetica', 'FontSize', 12, 'FontWeight', 'bold', 'Color', [0.12 0.12 0.12]);

    axL = axes('Parent', fig);
    hold(axL, 'on');
    set(axL, 'Position', [0.16 0.13 0.78 0.20], ...
        'YColor', 'none', ...
        'YLim', [0.5, size(Data,2) + 0.5], ...
        'XColor', 'none', ...
        'XLim', axI.XLim, ...
        'Box', 'off', ...
        'FontName', 'Helvetica');

    barHdlI = bar(axI, pCount, 0.72);
    barHdlI.EdgeColor = 'none';
    barHdlI.FaceColor = 'flat';
    for i = 1:length(pType)
        active_sets = logical(pBool(pType(i), :));
        barHdlI.CData(i, :) = get_intersection_color(currentColors(active_sets, :));
    end

    text(axI, 1:length(pType), pCount, string(pCount), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontName', 'Helvetica', ...
        'FontSize', 14.5, ...
        'Color', lineColor);

%     axI.YGrid = 'on';
    axI.GridColor = [0.90 0.91 0.93];
    axI.GridAlpha = 1;
    axI.Layer = 'top';

    for i = 1:size(Data, 2)
        fill(axL, axI.XLim([1,2,2,1]), [-0.5, -0.5, 0.5, 0.5] + i, stripeColor(mod(i + 1, 2) + 1, :), ...
            'EdgeColor', 'none');
    end

    [tX, tY] = meshgrid(1:length(pType), 1:size(Data, 2));
    plot(axL, tX(:), tY(:), 'o', 'Color', dotColor, 'MarkerFaceColor', dotColor, 'MarkerSize', 8);

    for i = 1:size(Data, 2)
        text(axL, 0, i, setName{sInd(i)}, ...
            'HorizontalAlignment', 'right', ...
            'VerticalAlignment', 'middle', ...
            'FontName', 'Helvetica', ...
            'FontSize', 10.5, ...
            'FontWeight', 'bold', ...
            'Color', currentColors(sInd(i), :));
    end

    for i = 1:length(pType)
        tY_v = find(pBool(pType(i), :));
        oY = zeros(size(tY_v));
        for j = 1:length(tY_v)
            oY(j) = find(sType == tY_v(j));
        end
        tX_v = i .* ones(size(tY_v));
        active_sets = logical(pBool(pType(i), :));
        active_color = get_intersection_color(currentColors(active_sets, :));
        plot(axL, tX_v(:), oY(:), '-o', ...
            'Color', active_color, ...
            'MarkerEdgeColor', 'none', ...
            'MarkerFaceColor', active_color, ...
            'MarkerSize', 8, ...
            'LineWidth', 1.8);
    end

    exportgraphics(fig, [out_base '.pdf'], 'ContentType', 'vector');
    close(fig);
end

function mixed_color = get_intersection_color(color_block)
    if size(color_block, 1) == 1
        mixed_color = 0.82 * color_block + 0.18 * [1 1 1];
    else
        mixed_color = mean(color_block, 1);
        mixed_color = 0.90 * mixed_color + 0.10 * [1 1 1];
    end
end

function out = lighten_colors(in, amount)
    out = in + (1 - in) * amount;
    out = min(max(out, 0), 1);
end

function create_model_legend(all_model_names, all_model_colors, out_base)
    fig_legend = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 18 2.4]);
    hold on;
    h_leg_patches = gobjects(1, numel(all_model_names));
    for k = 1:numel(all_model_names)
        h_leg_patches(k) = patch(NaN, NaN, all_model_colors(k, :), ...
            'EdgeColor', 'none', 'FaceAlpha', 0.62);
    end

    legend(h_leg_patches, all_model_names, ...
        'Interpreter', 'none', ...
        'Orientation', 'horizontal', ...
        'NumColumns', numel(all_model_names), ...
        'FontName', 'Helvetica', ...
        'FontSize', 14.5, ...
        'Box', 'off');

    axis off;
    set(gca, 'Position', [0 0 1 1]);
    exportgraphics(fig_legend, [out_base '.pdf'], 'ContentType', 'vector');
    close(fig_legend);
end
