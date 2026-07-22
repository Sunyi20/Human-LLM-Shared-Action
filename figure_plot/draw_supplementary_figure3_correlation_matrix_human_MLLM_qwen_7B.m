base_dir = pwd;
variable_dir = fullfile(base_dir,'variables');
%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

load(fullfile('data/Human/human_odd_one_out_spose_embedding_sorted_final.txt'));
spose_embedding_human = human_odd_one_out_spose_embedding_sorted_final;

num_actions = 256;
videos_per_action = 3;
excluded_action_idx = 83;
excluded_indices = (excluded_action_idx - 1) * videos_per_action + (1:videos_per_action);

all_indices = 1:(num_actions * videos_per_action);
valid_indices = setdiff(all_indices, excluded_indices);

spose_valid = spose_embedding_human(valid_indices, :);
num_valid_actions = num_actions - 1; 

spose_embedding_human_1 = zeros(num_valid_actions, size(spose_valid, 2));
spose_embedding_human_2 = zeros(num_valid_actions, size(spose_valid, 2));
spose_embedding_human_3 = zeros(num_valid_actions, size(spose_valid, 2));

valid_action_idx = 0;
for action_idx = 1:num_actions
    if action_idx == excluded_action_idx
        continue;
    end
    
    valid_action_idx = valid_action_idx + 1;

    video_start = (action_idx - 1) * videos_per_action + 1;
 
    if action_idx > excluded_action_idx
        offset = videos_per_action;
        adjusted_start = video_start - offset;
    else
        adjusted_start = video_start;
    end

    spose_embedding_human_1(valid_action_idx, :) = spose_valid(adjusted_start, :);
    spose_embedding_human_2(valid_action_idx, :) = spose_valid(adjusted_start + 1, :);
    spose_embedding_human_3(valid_action_idx, :) = spose_valid(adjusted_start + 2, :);
end

load(fullfile('data/MLLM_qwen_7B/qwen_7B_VL_spose_embedding_sorted_final.txt'));
spose_embedding = qwen_7B_VL_spose_embedding_sorted_final;

T_indices = readtable('data/folder_list/folder_list_human_odd_one_out_overlap.txt', 'FileType', 'text', 'ReadVariableNames', false);

action_indices = T_indices.Var1 + 1;
spose_embedding = spose_embedding(action_indices, :);

dim_human = 30;
dim = 30;

labels_human = 1:dim_human;
labels = 1:dim;

c1 = corr(spose_embedding_human_1,spose_embedding);
c2 = corr(spose_embedding_human_2,spose_embedding);
c3 = corr(spose_embedding_human_3,spose_embedding);

c = (c1 + c2 + c3)/3;
[~,ii] = max(c);
for i = 2:dim
    if any(ii(1:i-1)==ii(i))
        ii(i) = 100; 
    end
end
[~,si] = sort(ii);


c_adapted = c(:,si); 
fig = figure('Position',[300 1 800 700]);
imagesc(c(:,si))
caxis([-1 1]);
axis equal;
axis tight;
ax = gca;

ax.YTick = 1:dim_human;
ax.YTickLabels = labels_human;
set(ax, 'YTickLabel', ax.YTickLabels, 'FontSize', 18);
ax.XTick = 1:dim;
ax.XTickLabels = labels(si);
ax.XTickLabelRotation = 90;
set(ax, 'XTickLabel', ax.XTickLabels, 'FontSize', 18);

colormap(slanCM(102));
colorbar;
cbar = colorbar;
cbar.FontSize = 20;
set(gcf,'Units','centimeters')
screenposition = get(gcf,'Position');
set(gcf,'PaperPosition',[0 0 screenposition(3:4)])
set(gcf,'PaperSize',screenposition(3:4))

ylabel('Human Dimension', 'FontSize', 20);
xlabel('Qwen2.5-VL-7B Dimension', 'FontSize', 20);
exportgraphics(fig, 'reordered_corr_matrix_human_mllm_qwen_7B.pdf', 'ContentType', 'vector');
