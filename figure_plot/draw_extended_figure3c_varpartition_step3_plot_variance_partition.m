clear; clc; close all;

model_files = {
    'LLM_qwen_7B_varpart_results.mat',
    'LLM_deepseek_varpart_results.mat',
    'MLLM_qwen_7B_varpart_results.mat',
    'MLLM_qwen_72B_varpart_results.mat'
};

model_names = {
    'Qwen2.5-7B', 
    'Deepseek-R1', 
    'Qwen2.5-VL-7B', 
    'Qwen2.5-VL-72B'
};

num_models = length(model_files);
vp_avg_all = zeros(7, num_models);

base_dir = 'data/varpartition/';

for i = 1:num_models
    file_name = fullfile(base_dir, model_files{i});
    if isfile(file_name)
        data = load(file_name);
        vp_avg_all(:, i) = data.varpart_results.pred;
    else
        fprintf('Warning %s\n', file_name);
    end
end

nc1_all = zeros(1, num_models);
nc2_all = zeros(1, num_models);

plot_varpart_stacked(vp_avg_all, model_names, nc1_all, nc2_all);

fig = gcf;
fig.Position = [100, 100, 400, 600];  
set(gcf, 'Color', 'w'); 
drawnow;

if ~exist('figures', 'dir')
    mkdir('figures');
end

output_filename_stacked = 'figures/variance_partition_comparison_LLMs.pdf';

exportgraphics(gcf, output_filename_stacked, 'ContentType', 'vector');

fprintf('PDF saved to: %s\n', output_filename_stacked);


function [] = plot_varpart_stacked(vp_avg, tick_labels, nc1, nc2)

color1 = [150, 159, 214] / 255; % Activity (Model1)
color2 = [146, 203, 210] / 255; % Category (Model2)
color3 = [200, 217, 163] / 255; % Target (Model3)

color12 = (color1 + color2) / 2;
color13 = (color1 + color3) / 2;
color23 = (color2 + color3) / 2;
color123 = (color1 + color2 + color3) / 3;

colors_ordered_by_data = [
    color123;   % S123
    color12;    % S12
    color23;    % S23
    color13;    % S13
    color1;     % U1
    color2;     % U2
    color3      % U3
];

order_idx = [1, 2, 3, 4, 5, 6, 7]; 
legendstr = {'All', 'Activity+Category', 'Category+Target', 'Activity+Target', 'Activity', 'Category', 'Target'};

vp_plot = vp_avg(order_idx, :);
col_plot = colors_ordered_by_data(order_idx, :);

vp_plot(vp_plot < 0) = 0;

figure;
hold on;

if any(nc2 > 0)
    num_bars = numel(tick_labels);
    for i = 1:num_bars
        rectangle('Position', [i-0.4, nc1(i), 0.8, nc2(i)-nc1(i)], ...
                  'FaceColor', [0.85 0.85 0.85], 'EdgeColor', 'none');
    end
end

b = bar(vp_plot', 0.5, 'stacked', 'FaceColor', 'flat');

ylim_max = max(sum(vp_plot, 1)) * 1.2; 
if any(nc2 > 0)
    ylim_max = max([ylim_max, max(nc2) * 1.1]);
end
if ylim_max == 0
    ylim_max = 1; 
end
ylim([0, ylim_max]); 
xlim([0.5 numel(tick_labels) + 0.5]);

plot(xlim, [0 0], 'k-');

for c = 1:7
    b(c).LineWidth = 1.3;
    b(c).CData = col_plot(c, :);
end

box off;
set(gca, 'FontSize', 12);
set(gca, 'TickLength', [0.0001 0.0001]);
set(gca, 'XTick', 1:numel(tick_labels));
set(gca, 'XTickLabel', tick_labels);
xtickangle(30); 
ax = gca;
ax.XAxis.TickLabelInterpreter = 'none';

ylabel('Variance explained (Spearman''s \rho_A^2)');

lgd = legend(b, legendstr, 'Location', 'none'); 
legend boxoff;

lgd.Position = [0.75, 0.75, 0.09, 0.15];

lgd.FontSize = 11;

hold off;

end