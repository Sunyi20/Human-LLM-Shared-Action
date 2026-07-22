base_dir = pwd;

acc_human = 59.90;  
acc_LLM_qwen_7B = 64.51; 
acc_LLM_deepseek = 72.96;
acc_MLLM_qwen_7B = 56.34; 
acc_MLLM_qwen_72B = 61.69; 

chance_level_human = 18.80; 
lower_bound_human  = 16.28; 
upper_bound_human  = 21.48; 

chance_level_LLM_qwen_7B = 18.23;
lower_bound_LLM_qwen_7B  = 14.65;
upper_bound_LLM_qwen_7B  = 22.25;

chance_level_LLM_deepseek = 19.33;
lower_bound_LLM_deepseek  = 15.49;
upper_bound_LLM_deepseek  = 23.38;

chance_level_MLLM_qwen_7B = 17.54;
lower_bound_MLLM_qwen_7B  = 14.08;
upper_bound_MLLM_qwen_7B  = 21.13;

chance_level_MLLM_qwen_72B = 18.79;
lower_bound_MLLM_qwen_72B  = 14.93;
upper_bound_MLLM_qwen_72B  = 22.54;


customColor1 = [.2 .6 .4];   % LLM_qwen_7B
customColor2 = [.6 .2 .4];   % MLLM_qwen_72B
customColor3 = [.2 .3 .8];   % Human
customColor4 = [.9 .6 .1];   % MLLM_qwen_7B
customColor5 = [.2 .3 .4];   % LLM_deepseek

x_human         = 8;
x_llm_deepseek  = 22;
x_llm_qwen_7B   = 36;
x_mllm_qwen_7B  = 50;
x_mllm_qwen_72B = 64;
bw = 9; % BarWidth
bh = 4; % bar half width for chance line

fig = figure('Position',[800 800 500 800],'color','w');
hold on;

bar(x_human,         acc_human,         'FaceColor', customColor3, 'EdgeColor', 'none', 'BarWidth', bw);
bar(x_llm_deepseek,  acc_LLM_deepseek,  'FaceColor', customColor5, 'EdgeColor', 'none', 'BarWidth', bw);
bar(x_llm_qwen_7B,   acc_LLM_qwen_7B,   'FaceColor', customColor1, 'EdgeColor', 'none', 'BarWidth', bw);
bar(x_mllm_qwen_7B,  acc_MLLM_qwen_7B,  'FaceColor', customColor4, 'EdgeColor', 'none', 'BarWidth', bw);
bar(x_mllm_qwen_72B, acc_MLLM_qwen_72B, 'FaceColor', customColor2, 'EdgeColor', 'none', 'BarWidth', bw);

plot([x_human-bh,         x_human+bh],         [chance_level_human,         chance_level_human],         '-r', 'LineWidth', 2);
errorbar(x_human,         chance_level_human,         (upper_bound_human-lower_bound_human)/2,                 'Color', [0 0 0], 'LineWidth', 2, 'LineStyle', 'none');

plot([x_llm_deepseek-bh,  x_llm_deepseek+bh],  [chance_level_LLM_deepseek,  chance_level_LLM_deepseek],  '-r', 'LineWidth', 2);
errorbar(x_llm_deepseek,  chance_level_LLM_deepseek,  (upper_bound_LLM_deepseek-lower_bound_LLM_deepseek)/2,   'Color', [0 0 0], 'LineWidth', 2, 'LineStyle', 'none');

plot([x_llm_qwen_7B-bh,   x_llm_qwen_7B+bh],   [chance_level_LLM_qwen_7B,   chance_level_LLM_qwen_7B],   '-r', 'LineWidth', 2);
errorbar(x_llm_qwen_7B,   chance_level_LLM_qwen_7B,   (upper_bound_LLM_qwen_7B-lower_bound_LLM_qwen_7B)/2,     'Color', [0 0 0], 'LineWidth', 2, 'LineStyle', 'none');

plot([x_mllm_qwen_7B-bh,  x_mllm_qwen_7B+bh],  [chance_level_MLLM_qwen_7B,  chance_level_MLLM_qwen_7B],  '-r', 'LineWidth', 2);
errorbar(x_mllm_qwen_7B,  chance_level_MLLM_qwen_7B,  (upper_bound_MLLM_qwen_7B-lower_bound_MLLM_qwen_7B)/2,   'Color', [0 0 0], 'LineWidth', 2, 'LineStyle', 'none');

plot([x_mllm_qwen_72B-bh, x_mllm_qwen_72B+bh], [chance_level_MLLM_qwen_72B, chance_level_MLLM_qwen_72B], '-r', 'LineWidth', 2);
errorbar(x_mllm_qwen_72B, chance_level_MLLM_qwen_72B, (upper_bound_MLLM_qwen_72B-lower_bound_MLLM_qwen_72B)/2, 'Color', [0 0 0], 'LineWidth', 2, 'LineStyle', 'none');

baseline_y = 25; 

text(x_human,         baseline_y, 'Human',          'Rotation', 90, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Color', 'white', 'FontSize', 22);
text(x_llm_deepseek,  baseline_y, 'DeepSeek-R1',    'Rotation', 90, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Color', 'white', 'FontSize', 22);
text(x_llm_qwen_7B,   baseline_y, 'Qwen2.5-7B',     'Rotation', 90, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Color', 'white', 'FontSize', 22);
text(x_mllm_qwen_7B,  baseline_y, 'Qwen2.5-VL-7B',  'Rotation', 90, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Color', 'white', 'FontSize', 22);
text(x_mllm_qwen_72B, baseline_y, 'Qwen2.5-VL-72B', 'Rotation', 90, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Color', 'white', 'FontSize', 22);
text(x_human, 15, 'chance', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'red', 'FontSize', 18);

xlim([0, 72]);
ylim([0, 80]);
ylabel('Categorization performance (%)', 'FontSize', 34);

hax = gca;
set(gca, 'FontSize', 22);
hax.TickDir = 'both';
hax.XTick = [];
hax.XColor = [0 0 0];
hax.YColor = [0 0 0];
hax.LineWidth = 1.5;
hax.Box = 'off';

hold off;
%% Save Figure
output_dir = fullfile(base_dir, 'figures');
if ~exist(output_dir, 'dir')
    [success, msg] = mkdir(output_dir);
    if ~success
        error('Cannot Create: %s', msg);
    end
end
out_path = fullfile(output_dir, 'classification_results_all.pdf');
exportgraphics(fig, out_path, 'ContentType', 'vector');


