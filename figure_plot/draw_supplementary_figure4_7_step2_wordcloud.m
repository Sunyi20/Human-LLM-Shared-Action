clear;
clc;
close all;

base_dir = pwd;
data_dir = fullfile('data/dim_visualization');
output_dir = fullfile(data_dir, 'wordclouds_MLLM_qwen_72B');

addpath(base_dir);
addpath(genpath(fullfile(base_dir, 'helper_functions')));

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

load(fullfile(data_dir, 'dimlabel_answers_MLLM_qwen_72B.mat'));

h = fopen(fullfile('helper_functions/colors.txt'));
col = zeros(0, 3);
while 1
    l = fgetl(h);
    if l == -1, break, end
    col(end+1, :) = reshape(sscanf(l(2:end).', '%2x'), 3, []).' / 255; % hex2rgb
end
fclose(h);

colors = repmat(col, ceil(66 / size(col, 1)), 1); 
num_dimensions = size(dimlabel_answers_qwen, 2); 
dimlabel_words = cell(1, num_dimensions);
dimlabel_num_occur = cell(1, num_dimensions);

for i_dim = 1:num_dimensions
    str = strings;
    for i_sub = 1:size(dimlabel_answers_qwen, 1)
        s = strsplit(dimlabel_answers_qwen{i_sub, i_dim}, ',');
        for k = 1:length(s)
            s{k} = strtrim(s{k});
        end
        str(end+1:end+length(s)) = string(s);
    end
    str(1) = [];
    str = str';
    
    [dimlabel_words{i_dim}, ~, idx] = unique(str);
    dimlabel_num_occur{i_dim} = histcounts(idx, numel(dimlabel_words{i_dim}));
end

for i_dim = 1:num_dimensions
    fig = figure('Visible', 'off', 'Color', 'white');
    if isempty(dimlabel_words{i_dim})
        close(fig);
        continue;
    end

    try
        ha = wordcloud(dimlabel_words{i_dim}, dimlabel_num_occur{i_dim});
        ha.Color = [0 0 0];
        ha.HighlightColor = colors(i_dim, :);
        output_filename = fullfile(output_dir, sprintf('wordcloud_dim_%d.pdf', i_dim));
        % exportgraphics(gca, output_filename, 'ContentType', 'vector', 'BackgroundColor', 'white');
        saveas(gcf, output_filename);
        
    catch ME
        fprintf('    -> Dimension %d Error: %s\n', i_dim, ME.message);
    end
    close(fig);
end
