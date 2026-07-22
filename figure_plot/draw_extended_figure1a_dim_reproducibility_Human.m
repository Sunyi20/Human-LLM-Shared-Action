clear all;
% run this script from where it is located
base_dir = pwd;

%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

% load embedding
spose_embedding = load('data/Human/human_odd_one_out_spose_embedding_sorted_final.txt');
dim = size(spose_embedding,2);
dosave = 1;

refdir = fullfile('data/Human/reference_human_odd_one_out_spose');
for i_model = 1:65
    fn = dir(fullfile(refdir,sprintf('s%02i',i_model),'*.txt'));
    fn = fullfile(fn(end).folder,fn(end).name);
    tmp = load(fn);
    % remove empty dimensions
    tmp2 = tmp(:,any(tmp>0.1));
    reference_models{i_model,1} = tmp2;
    n_dim_reference(i_model) = size(reference_models{i_model},2);
end

% Correlate dimensions (this slightly overestimates the performance, given
% that each dimension can be picked several times, but there is no other
% way - otherwise some dimensions would go unmatched)
for i_model = 1:65
    % Select only the 65 specified actions from both embeddings before correlation
    % reference_model_subset = reference_models{i_model}(action_indices, :);
    reproducibility(:,i_model) = max(corr(spose_embedding, reference_models{i_model}), [], 2);
end

% test split-half prediction
for i_model = 1:65
    spose_embedding_subset = spose_embedding;
    reference_model_subset = reference_models{i_model};

    [~,maxind(:,i_model)] = max(corr(spose_embedding_subset(1:2:end,:),reference_model_subset(1:2:end,:)),[],2);
    [~,maxind2(:,i_model)] = max(corr(spose_embedding_subset(2:2:end,:),reference_model_subset(2:2:end,:)),[],2);
    c1 = corr(spose_embedding_subset(1:2:end,:),reference_model_subset(1:2:end,:));
    c2 = corr(spose_embedding_subset(2:2:end,:),reference_model_subset(2:2:end,:));
    for i = 1:dim, tmp1(i,i_model) = c1(i,maxind2(i,i_model)); tmp2(i,i_model) = c2(i,maxind(i,i_model)); end
end

epsilon = 1e-6;
reproducibility_clipped = min(max(reproducibility, -1 + epsilon), 1 - epsilon);

% fisher-z transform (atanh) and average
mean_reproducibility_z = mean(atanh(reproducibility_clipped), 2);
reproducibility_ci95_z = 1.96 * std(atanh(reproducibility_clipped), [], 2) / sqrt(65);

upper_bound = tanh(mean_reproducibility_z + reproducibility_ci95_z);
lower_bound = tanh(mean_reproducibility_z - reproducibility_ci95_z);
mean_reproducibility = tanh(mean_reproducibility_z);

fig = figure('Position',[800 800 500 400],'color','none');
hold on

x = [1:dim dim:-1:1];
y = [lower_bound' upper_bound(end:-1:1)'];

% 绘制置信区间
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';

% 绘制均值线和散点
plot(mean_reproducibility,'k','linewidth',2.0)
plot(reproducibility,'o','MarkerFaceColor',[.2 .3 .8],'MarkerEdgeColor','none','MarkerSize',3)

ylim([0 1])
xlim([0 dim+1])

set(gca,'FontSize',15) 
ylabel('Reproducibility score', 'FontSize', 18);
xlabel({'Human embedding dimension index'}, 'FontSize', 18);

hax = gca;
hax.LineWidth = 1.5; 

imageSize = [500 400];
set(fig, 'PaperUnits', 'inches', 'PaperSize', imageSize/100, 'PaperPosition', [0 0 imageSize/100]);

if dosave
    % 去掉 '-bestfit'，直接按设定的 PaperSize 打印，这样字体比例就完美同步了
    print(fig, 'dim_reproducibility_Human', '-dpdf');
end

close(fig);

% Test correlation between rank of reliability and dimension number
[~,reproducibility_ind] = sort(mean_reproducibility,'descend');
r_rank = corr((1:dim)',reproducibility_ind);

% run 100000 permutations
rng(1)
[~,perm] = sort(rand(dim,100000));
r_rank_perm = corr(perm,reproducibility_ind);
p = mean([r_rank_perm;r_rank] >= r_rank);

% run 1000 bootstrap samples for confidence intervals
rng(2)
rnd = randi(dim,dim,1000);
for i = 1:1000
    r_rank_boot(:,i) = corr(rnd(:,i),reproducibility_ind(rnd(:,i)));
end
r_rank_ci95_lower = tanh(atanh(r_rank) - 1.96*std(atanh(r_rank_boot),[],2));
r_rank_ci95_upper = tanh(atanh(r_rank) + 1.96*std(atanh(r_rank_boot),[],2));