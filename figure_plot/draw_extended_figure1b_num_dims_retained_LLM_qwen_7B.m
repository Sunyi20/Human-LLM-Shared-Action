base_dir = pwd;
%% Add relevant toolboxes
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

% load similarity computed from embedding (using embedding2sim.m)
load(fullfile('data/LLM_qwen_7B/qwen_7B_spose_similarity.mat'))
% spose_sim=embedding2sim(spose_embedding30);
dissim = 1-similarity_matrix;
% load test set
spose_embedding30= load(fullfile('data/LLM_qwen_7B/qwen_7B_spose_embedding_sorted_final.txt'));
% get dot product (i.e. proximity)
dot_product30= spose_embedding30*spose_embedding30';

% load 10% validation (i.e., test) data
triplet_testdata30 = load(fullfile('data/LLM_qwen_7B/test_triplets_llm.txt'))+1; % 0 index -> 1 index
%% in the training and test datasets, the order is still wrong, let's change it


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate how much variance can be explained in the test set %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
behav_predict = zeros(length(triplet_testdata30),1);
behav_predict_prob = zeros(length(triplet_testdata30),1);
rng(42) % for reproducibility
for i = 1:length(triplet_testdata30)
    sim(1) = dot_product30(triplet_testdata30(i,1),triplet_testdata30(i,2));
    sim(2) = dot_product30(triplet_testdata30(i,1),triplet_testdata30(i,3));
    sim(3) = dot_product30(triplet_testdata30(i,2),triplet_testdata30(i,3));
    [m,mi] = max(sim); % people are expected to choose the pair with the largest dot product
    if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
    behav_predict(i,1) = mi;
    behav_predict_prob(i,1) = exp(sim(mi))/sum(exp(sim)); % get choice probability
end
% get overall prediction (predict choice == 1)
behav_predict_acc = 100*mean(behav_predict==1);
% get prediction for each object
for i_obj = 1:355
    behav_predict_obj(i_obj,1) = 100*mean(behav_predict(any(triplet_testdata30==i_obj,2))==1);
    % this below gives us the predictability of each object on average
    % (i.e. how difficult it is expected to predict choices with it irrespective of other objects)
    behav_predict_obj_prob(i_obj,1) = 100*mean(behav_predict_prob(any(triplet_testdata30==i_obj,2)));
end
% get 95% CI for this value across objects
behav_predict_acc_ci95 = 1.96*std(behav_predict_obj)/sqrt(355);

%% Calculate how the behavioral prediction changes when eliminating dimensions with small weight

dosave = 1;

% do this for each obj separately, i.e. get index for each object across dimensions, then eliminate one by one

fn1 = fullfile('data/LLM_qwen_7B/qwen_7B_spose_similarity_reduced.mat');
fn2 = fullfile('data/LLM_qwen_7B/qwen_7B_spose_embedding30_reduced.mat');

if ~exist(fn1,'file') || ~exist(fn2','file')
    [~,embedding_sortind] = sort(spose_embedding30,2);
    disp('Getting reduced versions of embeddings and converting them to similarity.')
    disp('This takes about 10-15min on a regular laptop but only needs to be run once.')
    for i_dim = 1:30
        fprintf('.')
        % make 30 reduced versions, make 30 reduced similarity matrices
        if i_dim == 1
            qwen_7B_spose_embedding30_reduced{i_dim,1} = spose_embedding30;
        else
            qwen_7B_spose_embedding30_reduced{i_dim,1} = qwen_7B_spose_embedding30_reduced{i_dim-1,1};
        end
        for i = 1:355
            qwen_7B_spose_embedding30_reduced{i_dim,1}(i,embedding_sortind(i,i_dim)) = 0;
        end
        qwen_7B_spose_similarity_reduced{i_dim,1} = embedding2sim(qwen_7B_spose_embedding30_reduced{i_dim,1});
        
    end
    fprintf('\n')
    save(fn1,'qwen_7B_spose_similarity_reduced')
    save(fn2,'qwen_7B_spose_embedding30_reduced')
else
    load(fn1)
    load(fn2)
end

clear sim
for i_dim = 1:30
    rng(42) % for reproducibility
    behav_predict = zeros(length(triplet_testdata30),1);
    dot_product30_reduc = qwen_7B_spose_embedding30_reduced{i_dim}*qwen_7B_spose_embedding30_reduced{i_dim}';
    for i = 1:length(triplet_testdata30)
        sim(1) = dot_product30_reduc(triplet_testdata30(i,1),triplet_testdata30(i,2));
        sim(2) = dot_product30_reduc(triplet_testdata30(i,1),triplet_testdata30(i,3));
        sim(3) = dot_product30_reduc(triplet_testdata30(i,2),triplet_testdata30(i,3));
        [m,mi] = max(sim);
        if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
        behav_predict(i,1) = mi;
    end
    % get overall prediction (predict choice == 1)
    behav_predict_acc_reduc(i_dim) = 100*mean(behav_predict==1);
    % get prediction for each object
    for i_obj = 1:355
        behav_predict_obj_reduc(i_obj,i_dim) = 100*mean(behav_predict(any(triplet_testdata30==i_obj,2))==1);
    end
    % get standard error for this value across objects
    behav_predict_acc_reduc_ci95(i_dim,1) = 1.96* std(behav_predict_obj_reduc(:,i_dim))/sqrt(355);
end

% now reverse it all
behav_predict_acc_reduc = behav_predict_acc_reduc(end:-1:1);
behav_predict_obj_reduc = behav_predict_obj_reduc(:,end:-1:1);
behav_predict_acc_reduc_ci95 = behav_predict_acc_reduc_ci95(end:-1:1);

cutoff95 = (0.95*behav_predict_acc-100/3)+100/3;
cutoff99 = (0.99*behav_predict_acc-100/3)+100/3;

mindim = find(behav_predict_acc_reduc>cutoff95,1,'first')-1; % -1 because we need to start counting at 0
maxdim = find(behav_predict_acc_reduc<cutoff99,1,'last')-1;
fprintf('We need between %i and %i dimensions to reach 95-99%% performance in predicting individual trials.\n',mindim,maxdim)

% compare similarity matrices
for i_dim = 1:30
    r_reduc(i_dim) = corr(squareformq(similarity_matrix),squareformq(qwen_7B_spose_similarity_reduced{i_dim}));
end

% reverse r_reduc
r_reduc = r_reduc(end:-1:1);

mindim2 = find(r_reduc.^2>0.95,1,'first')-1; % -1 because we need to start counting at 0
maxdim2 = find(r_reduc.^2<0.99,1,'last')-1;
fprintf('We need between %i and %i dimensions to explain 95-99%% variance in similarity.\n',mindim2,maxdim2)


fig = figure('Position',[500 500 500 500],'color','none');
cutoff95 = (0.95*behav_predict_acc-100/3)+100/3;
cutoff99 = (0.99*behav_predict_acc-100/3)+100/3;
% also, add noise ceiling
hold on
x = [0 67 67 0];
y = [cutoff95 cutoff95 cutoff99 cutoff99];
hc = patch(x,y,[0.7 0.7 0.7]);
hc.EdgeColor = 'none';
% hc.FaceAlpha = 0.3;
x = [mindim maxdim maxdim mindim];
y = [70 70 0 0];
hcb = patch(x,y,[.2 .6 .4]);
hcb.EdgeColor = 'none';
% hcb.FaceAlpha = 0.5;
plot(0:30,[behav_predict_acc_reduc behav_predict_acc],'k','LineWidth',3)
plot([0 30],[100/3 100/3],'--r')
text(50, 35, 'chance', 'Rotation', 0, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 14);
xlim([0 30])
ylim([0 70])
xlabel('Number of dimensions retained', 'FontSize', 20)
ylabel('Accuracy (%)', 'FontSize', 20)
title('Prediction of LLM behavior', 'FontSize', 20)

ax = gca;
set(ax, 'FontSize', 17);

if dosave
    exportgraphics(fig, 'acc_vs_dims_retained_LLM_qwen_7B.pdf', 'ContentType', 'vector');
end
