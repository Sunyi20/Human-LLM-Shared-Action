% run this script from where it is located
base_dir = pwd;
%% Add relevant toolboxes
variable_dir = fullfile('variables');

addpath(base_dir)
addpath(genpath(fullfile('helper_functions')))

index_file_path = 'data/folder_list/folder_list_full_sample_41.txt';
T_indices = readtable(index_file_path, 'FileType', 'text', 'ReadVariableNames', false);
action_indices = T_indices.Var1 + 1;

% load embedding
spose_embedding = load(fullfile('data/MLLM_qwen_72B/qwen_72B_spose_embedding_sorted_full_sample.txt'));

dot_product = spose_embedding*spose_embedding';

% load RDM
load(fullfile('data/MLLM_qwen_72B/RDM_48_MLLM_qwen_72B.mat'));
RDM41_triplet = RDM_triplet(action_indices, action_indices);

dosave = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now compare similarity from model to similarity in behavior %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic
esim = exp(dot_product);
n_objects = size(esim, 1); 
cp = zeros(n_objects, n_objects);
ctmp = zeros(1, n_objects);

for i = 1:n_objects
    for j = i+1:n_objects
        ctmp = zeros(1, n_objects);
        for k = 1:n_objects
            if k == i || k == j, continue, end
            ctmp(k) = esim(i,j) / (esim(i,j) + esim(i,k) + esim(j,k));
        end
        cp(i,j) = sum(ctmp);
    end
end
toc

cp = cp/(n_objects); 
cp = cp+cp';
cp(logical(eye(size(cp)))) = 1;

spose_sim41 = cp;

r41 = corr(squareformq(spose_sim41),squareformq(1-RDM41_triplet))

[R, p_value] = corr(squareformq(spose_sim41), squareformq(1-RDM41_triplet), 'Type', 'Pearson');

rng(2)
rnd = randi(nchoosek(n_objects,2),nchoosek(n_objects,2),1000);
c1 = squareformq(spose_sim41);
c2 = squareformq(1-RDM41_triplet);
r41_boot = zeros(1, 1000);
for i = 1:1000
    r41_boot(:,i) = corr(c1(rnd(:,i)),c2(rnd(:,i)));
end
r41_ci95_lower = tanh(atanh(r41) - 1.96*std(atanh(r41_boot),[],2)) 
r41_ci95_upper = tanh(atanh(r41) + 1.96*std(atanh(r41_boot),[],2)) 

spose_sim41 = spose_sim41;
RDM41_triplet = RDM41_triplet;
r41 = r41;
r41_ci95_lower = r41_ci95_lower;
r41_ci95_upper = r41_ci95_upper;

disp('Creating publication-quality figure...');

figureWidth = 1800; 
figureHeight = 610;
fig = figure('Position', [100, 100, figureWidth, figureHeight], ...
             'Color', 'white', ...
             'Units', 'pixels');

natureColors = {
    [0, 90, 171]/255;    
    [220, 50, 32]/255;   
    [0, 155, 118]/255;   
    [240, 180, 0]/255;   
    [128, 0, 128]/255;   
    [0, 0, 0]            
};

set(groot, 'DefaultAxesFontName', 'Arial');
set(groot, 'DefaultTextFontName', 'Arial');

ax1 = subplot(1, 3, 1);
imagesc(spose_sim41);
colormap(ax1, slanCM(102));
caxis([0 1]); 
axis equal tight;
box on;
set(ax1, 'FontSize', 20, 'LineWidth', 1.5, ...
         'TickDir', 'out', 'XColor', 'k', 'YColor', 'k');

text(0.5, -0.13, 'Predicted RSM', ...
     'Units', 'normalized', ...
     'HorizontalAlignment', 'center', ...
     'FontSize', 20, 'FontWeight', 'bold');

text(0.5, 1.05, sprintf('%d diverse actions', n_objects), ...
     'Units', 'normalized', ...
     'HorizontalAlignment', 'center', ...
     'FontSize', 20, 'FontWeight', 'normal');

ax2 = subplot(1, 3, 2);
measured_rsm = 1 - RDM41_triplet;
imagesc(measured_rsm);
colormap(ax2, slanCM(102));
caxis([0 1]);

axis equal tight;
box on;
set(ax2, 'FontSize', 20, 'LineWidth', 1.5, ...
         'TickDir', 'out', 'XColor', 'k', 'YColor', 'k');

text(0.5, -0.13, 'Measured RSM', ...
     'Units', 'normalized', ...
     'HorizontalAlignment', 'center', ...
     'FontSize', 20, 'FontWeight', 'bold');

cb = colorbar('Location', 'eastoutside');
cb.Position = [0.042, 0.28, 0.01, 0.54];
cb.FontSize = 20;
cb.Label.String = 'Similarity';
cb.Label.FontSize = 20;
cb.Label.FontWeight = 'bold';
cb.Label.Rotation = 90;
cb.Label.VerticalAlignment = 'bottom';

ax3 = subplot(1, 3, 3);

x_data = squareformq(spose_sim41);
y_data = squareformq(measured_rsm);

scatter(ax3, x_data, y_data, 35, ...
        'MarkerFaceColor', [.6 .2 .4], ...
        'MarkerFaceAlpha', 0.6, ...
        'MarkerEdgeColor', 'none');

hold on;

p = polyfit(x_data, y_data, 1);
y_fit = polyval(p, [0, 1]);
plot(ax3, [0, 1], y_fit, 'Color', natureColors{2}, ...
     'LineWidth', 2.5, 'LineStyle', '--');

plot(ax3, [0, 1], [0, 1], 'Color', [0.3, 0.3, 0.3], ...
     'LineWidth', 1.5, 'LineStyle', ':');

xlim([0 1]);
ylim([0 1]);
axis square;
box on;
grid on;
grid minor;
alpha(0.3);

set(ax3, 'FontSize', 20, 'LineWidth', 1.5, ...
         'TickDir', 'out', ...
         'GridAlpha', 0.2, 'MinorGridAlpha', 0.1);

xlabel('Predicted similarity', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('Measured similarity', 'FontSize', 20, 'FontWeight', 'bold');

stats_text = sprintf('r = %.2f [%.2f, %.2f]\\newlineP < 0.001\\newline\\color{red}-- \\color[rgb]{0.3,0.3,0.3}Linear fit', ...
                    r41, r41_ci95_lower, r41_ci95_upper);

text(0.05, 0.95, stats_text, ...
     'Units', 'normalized', ...
     'FontSize', 17, ...
     'FontWeight', 'bold', ...
     'BackgroundColor', [1, 1, 1, 0.8], ...
     'EdgeColor', 'k', ...
     'Margin', 2, ...
     'VerticalAlignment', 'top');


pos1 = get(ax1, 'Position');
pos2 = get(ax2, 'Position');
pos3 = get(ax3, 'Position');

new_width = 0.22;
new_height = 0.7;

set(ax1, 'Position', [0.082, 0.2, new_width, new_height]);

set(ax2, 'Position', [0.343, 0.2, new_width, new_height]);

set(ax3, 'Position', [0.63, 0.2, new_width, new_height]);

annotation('textbox', [0, 0.88, 1, 0.1], ...
           'String', 'Qwen2.5-VL-72B: Comparison of Predicted and Measured RSMs', ...
           'EdgeColor', 'none', ...
           'HorizontalAlignment', 'center', ...
           'FontSize', 22, ...
           'FontWeight', 'bold');

if ~exist('dosave', 'var') || dosave
    exportgraphics(fig, 'figures/RSM_Qwen2_5_VL_72B_41.pdf', ...
                   'ContentType', 'vector', ...
                   'Resolution', 400);

end
