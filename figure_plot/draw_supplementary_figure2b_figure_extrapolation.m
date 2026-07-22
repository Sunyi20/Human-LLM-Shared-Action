base_dir = pwd;
addpath(genpath(fullfile(base_dir,'helper_functions')))


load data/extrapolation/output_file.mat

for i = 1:12
    ind = (i-1)*4+(1:4);
    m_best_dim(i) = mean(best_dim(ind));
    dim_res{i} = best_dim(ind);
end
x_vals = [5, 7, 9, 11, 13, 16, 19, 20, 21, 23, 27, 30];

% extrapolate function using an exponential in the shape of
% a + b exp(-cx)

% parameter A: when dataset size is infinite and assuming a function that decays,
% this parameter reflects the dimensionality in the limit. Thus, this number 
% must be positive and larger than the current dimensionality
% (let's assume 100 = parameter of 1)
% parameter B: estimated to be negative (no growth but decay)
% parameter C: no clear estimate, so let's use 1
A = expcurvefit(x_vals,m_best_dim/100,[2 -2 1]);
Aest = A(1); Best = A(2); Cest = A(3);
Fitted = Aest + (Best.* exp(-Cest * [1:60 100]));

%% bootstrap

% we want to find the error of our curve fitting exercise

rng(1)
if ~exist('data/extrapolation/Fitted_boot_48.mat','file')
    
    for i = 1:12
        ind = (i-1)*4+(1:4);
        for j = 1:1000
            ind2 = ind(randi(4,4,1));
            me_best_dim_boot(i,j) = mean(best_dim(ind2));
        end
    end
    
    for j = 1000:-1:1
        Aboot(:,j) = expcurvefit(x_vals,me_best_dim_boot(:,j)'/100,[0.675 -0.67 0.09]);
    end
    
    for j = 1000:-1:1
        Fitted_boot(:,j) = Aboot(1,j) + (Aboot(2,j).* exp(-Aboot(3,j) * [1:60 100]));
    end
    
    save('data/extrapolation/Fitted_boot_48.mat','Fitted_boot')
    
else
    load('data/extrapolation/Fitted_boot_48.mat')
end

figure('Position',[450 100 900 650], 'Color', 'w');
hold on;
x_fit = 1:60;
y_lower = reshape(prctile(100*Fitted_boot(1:60,:)', 2.5), 1,[]);
y_upper = reshape(prctile(100*Fitted_boot(1:60,:)', 97.5), 1,[]);
fill_x = [x_fit, fliplr(x_fit)];
fill_y = [y_lower, fliplr(y_upper)];
h_ci = patch(fill_x, fill_y, [0.85 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
h_dist = distributionPlot(dim_res, 'xValues', x_vals, 'showMM', 0, ...
    'color',[0.8 0.9 0.95]); 
for i = 1:length(h_dist{1})
    if ishandle(h_dist{1}(i))
        set(h_dist{1}(i), 'EdgeColor', [0.4 0.6 0.8], 'LineWidth', 1);
    end
end

h_spread = plotSpread(dim_res, 'xValues', x_vals);
set(h_spread{1}, 'Marker', '.', 'MarkerSize', 15, 'Color',[0.6 0.2 0.2]); 
dim_means = zeros(1, length(x_vals));
dim_stds = zeros(1, length(x_vals));
for i = 1:length(dim_res)
    dim_means(i) = mean(dim_res{i});
    dim_stds(i) = std(dim_res{i});
end
h_error = errorbar(x_vals, dim_means, dim_stds, '-o', 'Color', [1 0 0], ...
    'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', [1 0 0], ...
    'MarkerEdgeColor',[1 0 0], 'CapSize', 5);

y_fit = 100 * Fitted(1:60);
h_fit = plot(x_fit, y_fit, '--', 'Color', [0.7 0 0], 'LineWidth', 2.5);

xpos = 5278093/1e5;
y_ref = 31:37;
n_dim_reference = [31,31,31,32,32,32,32,32,32,32,32,32,33,33,33,33,33,33,33,33,33,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,35,35,35,35,35,35,35,36,36,36,36,36,36,36,37,37,37,37]
d_ref = histc(n_dim_reference, y_ref);
h_ref_dist = distributionPlot({[y_ref' d_ref']}, 'xValues', xpos, 'showMM', 0, ...
    'color',[0.8 0.9 0.95], 'distWidth', 2);
set(h_ref_dist{1}, 'EdgeColor', [0.4 0.6 0.8], 'LineWidth', 1);
h_star = plot(xpos, mean(n_dim_reference), 'p', 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'MarkerSize', 12);
text(xpos + 1.5, mean(n_dim_reference), 'Chosen Dataset Size', 'FontSize', 12, 'FontWeight', 'bold');

ax = gca;
ax.YGrid = 'on';
ax.XGrid = 'off';
ax.GridLineStyle = '--';
ax.GridColor = [0.8 0.8 0.8];
ax.GridAlpha = 0.6;

ax.XTick =[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55];
ax.XLim =[0, max(ax.XTick) + 5];
ax.YTick = 0:10:50;
ax.YLim =[min(y_lower)-2, 50];

ax.FontSize = 14;
ax.FontName = 'Arial';
ylabel('Number of dimensions', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Dataset size (in 10k trials)', 'FontSize', 16, 'FontWeight', 'bold');

lgd = legend([h_spread{1}(1), h_error, h_fit, h_ci], ...
    {'Raw data', 'Mean \pm SD', 'Exponential Fit', '95% CI (Bootstrapped)'}, ...
    'Location', 'northwest', 'FontSize', 12);
lgd.EdgeColor =[0.6 0.6 0.6];
lgd.ItemTokenSize = [20, 18];

set(gcf, 'Renderer', 'painters');
print(gcf, '-dpdf', 'extrapolation.pdf', '-bestfit');