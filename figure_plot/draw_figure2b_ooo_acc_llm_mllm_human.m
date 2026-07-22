clear all;
base_dir = pwd;
addpath(base_dir)
addpath(genpath(fullfile(base_dir,'helper_functions')))

%% LLM_deepseek
% load embedding
spose_embedding30= load(fullfile('data/LLM_deepseek/deepseek_spose_embedding_sorted_final.txt'));
% get dot product (i.e. proximity)
dot_product30= spose_embedding30*spose_embedding30';

% load 10% validation (i.e., test) data
triplet_testdata = load(fullfile('data/LLM_deepseek/test_triplets_llm.txt'))+1; % 0 index -> 1 index
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate how much variance can be explained in the test set %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
behav_predict = zeros(length(triplet_testdata),1);
behav_predict_prob = zeros(length(triplet_testdata),1);
rng(42) % for reproducibility
for i = 1:length(triplet_testdata)
    sim(1) = dot_product30(triplet_testdata(i,1),triplet_testdata(i,2));
    sim(2) = dot_product30(triplet_testdata(i,1),triplet_testdata(i,3));
    sim(3) = dot_product30(triplet_testdata(i,2),triplet_testdata(i,3));
    [m,mi] = max(sim); % people are expected to choose the pair with the largest dot product
    if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
    behav_predict(i,1) = mi;
    behav_predict_prob(i,1) = exp(sim(mi))/sum(exp(sim)); % get choice probability
end
% get overall prediction (predict choice == 1)
behav_predict_acc_llm_deepseek = 100*mean(behav_predict==3)

num_permutations = 1000;
permuted_accuracies_llm_deepseek = zeros(1, num_permutations);
for i = 1:num_permutations
    permuted_label = randi([1, 3], 1, length(behav_predict));
    permuted_accuracies_llm_deepseek(i) = 100*mean(behav_predict==permuted_label');
end
% chance level
chance_level_llm_deepseek = mean(permuted_accuracies_llm_deepseek);
% chance level 95% CI
lower_bound_llm_deepseek = prctile(permuted_accuracies_llm_deepseek, 2.5);
upper_bound_llm_deepseek = prctile(permuted_accuracies_llm_deepseek, 97.5);
fprintf('Chance Level LLM: %.4f\n', chance_level_llm_deepseek);
fprintf('Approximate 95%% CI for Chance Level: [%.4f, %.4f]\n', lower_bound_llm_deepseek, upper_bound_llm_deepseek);


% get prediction for each object
for i_obj = 1:355
    behav_predict_obj(i_obj,1) = 100*mean(behav_predict(any(triplet_testdata==i_obj,2))==1);
    % this below gives us the predictability of each object on average
    % (i.e. how difficult it is expected to predict choices with it irrespective of other objects)
    behav_predict_obj_prob(i_obj,1) = 100*mean(behav_predict_prob(any(triplet_testdata==i_obj,2)));
end
% get 95% CI for this value across objects
behav_predict_acc_ci95_llm_deepseek = 1.96*std(behav_predict_obj)/sqrt(355);

%%%%%%%%%%%%%%%%%%%%%
% Get noise ceiling %
%%%%%%%%%%%%%%%%%%%%%
data_llm = load('data/LLM_deepseek/deepseek_noise_ceiling_results.mat');
probabilities_llm = data_llm.probabilities;
noise_ceiling_llm_deepseek = mean(probabilities_llm) * 100;
noise_ceiling_std_llm_deepseek = std(probabilities_llm) * 100;
noise_ceiling_ci95_llm_deepseek = 1.96 * noise_ceiling_std_llm_deepseek / sqrt(length(probabilities_llm));



%% MLLM_qwen_7B

% load embedding
spose_embedding30= load(fullfile('data/MLLM_qwen_7B/qwen_7B_VL_spose_embedding_sorted_final.txt'));
% get dot product (i.e. proximity)
dot_product30= spose_embedding30*spose_embedding30';

% load 10% validation (i.e., test) data
triplet_testdata = load(fullfile('data/MLLM_qwen_7B/test_triplets_mllm.txt'))+1; % 0 index -> 1 index
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate how much variance can be explained in the test set %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
behav_predict = zeros(length(triplet_testdata),1);
behav_predict_prob = zeros(length(triplet_testdata),1);
rng(42) % for reproducibility
for i = 1:length(triplet_testdata)
    sim(1) = dot_product30(triplet_testdata(i,1),triplet_testdata(i,2));
    sim(2) = dot_product30(triplet_testdata(i,1),triplet_testdata(i,3));
    sim(3) = dot_product30(triplet_testdata(i,2),triplet_testdata(i,3));
    [m,mi] = max(sim); % people are expected to choose the pair with the largest dot product
    if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
    behav_predict(i,1) = mi;
    behav_predict_prob(i,1) = exp(sim(mi))/sum(exp(sim)); % get choice probability
end
% get overall prediction (predict choice == 1)
behav_predict_acc_mllm_qwen_7B = 100*mean(behav_predict==3)

num_permutations = 1000;
permuted_accuracies_mllm_qwen_7B = zeros(1, num_permutations);
for i = 1:num_permutations
    permuted_label = randi([1, 3], 1, length(behav_predict));
    permuted_accuracies_mllm_qwen_7B(i) = 100*mean(behav_predict==permuted_label');
end
% chance level
chance_level_mllm_qwen_7B = mean(permuted_accuracies_mllm_qwen_7B);
% chance level 95% CI
lower_bound_mllm_qwen_7B = prctile(permuted_accuracies_mllm_qwen_7B, 2.5);
upper_bound_mllm_qwen_7B = prctile(permuted_accuracies_mllm_qwen_7B, 97.5);
fprintf('Chance Level MLLM Qwen 7B: %.4f\n', chance_level_mllm_qwen_7B);
fprintf('Approximate 95%% CI for Chance Level: [%.4f, %.4f]\n', lower_bound_mllm_qwen_7B, upper_bound_mllm_qwen_7B);


% get prediction for each object
for i_obj = 1:355
    behav_predict_obj(i_obj,1) = 100*mean(behav_predict(any(triplet_testdata==i_obj,2))==1);
    % this below gives us the predictability of each object on average
    % (i.e. how difficult it is expected to predict choices with it irrespective of other objects)
    behav_predict_obj_prob(i_obj,1) = 100*mean(behav_predict_prob(any(triplet_testdata==i_obj,2)));
end
% get 95% CI for this value across objects
behav_predict_acc_ci95_mllm_qwen_7B = 1.96*std(behav_predict_obj)/sqrt(355);

%%%%%%%%%%%%%%%%%%%%%
% Get noise ceiling %
%%%%%%%%%%%%%%%%%%%%%
data_mllm_qwen_7B = load('data/MLLM_qwen_7B/qwen_7B_VL_noise_ceiling_results.mat');
probabilities_mllm_qwen_7B = data_mllm_qwen_7B.probabilities;
noise_ceiling_mllm_qwen_7B = mean(probabilities_mllm_qwen_7B) * 100;
noise_ceiling_std_mllm_qwen_7B = std(probabilities_mllm_qwen_7B) * 100;
noise_ceiling_ci95_mllm_qwen_7B = 1.96 * noise_ceiling_std_mllm_qwen_7B / sqrt(length(probabilities_mllm_qwen_7B));



%% MLLM_qwen_72B
% load embedding
spose_embedding30= load(fullfile('data/MLLM_qwen_72B/qwen_72B_VL_spose_embedding_sorted_final.txt'));
% get dot product (i.e. proximity)
dot_product30= spose_embedding30*spose_embedding30';

% load 10% validation (i.e., test) data
triplet_testdata = load(fullfile('data/MLLM_qwen_72B/test_triplets_mllm.txt'))+1; % 0 index -> 1 index
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate how much variance can be explained in the test set %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
behav_predict = zeros(length(triplet_testdata),1);
behav_predict_prob = zeros(length(triplet_testdata),1);
rng(42) % for reproducibility
for i = 1:length(triplet_testdata)
    sim(1) = dot_product30(triplet_testdata(i,1),triplet_testdata(i,2));
    sim(2) = dot_product30(triplet_testdata(i,1),triplet_testdata(i,3));
    sim(3) = dot_product30(triplet_testdata(i,2),triplet_testdata(i,3));
    [m,mi] = max(sim); % people are expected to choose the pair with the largest dot product
    if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
    if mi == 1
        behav_predict(i,1) = 3;
    elseif mi == 2
        behav_predict(i,1) = 2;
    else
        behav_predict(i,1) = 1;
    end
    behav_predict_prob(i,1) = exp(sim(mi))/sum(exp(sim)); % get choice probability
end
% get overall prediction (predict choice == 1)
behav_predict_acc_mllm_qwen_72B = 100*mean(behav_predict==3)

num_permutations = 1000;
permuted_accuracies_mllm_qwen_72B = zeros(1, num_permutations);
for i = 1:num_permutations
    permuted_label = randi([1, 3], 1, length(behav_predict));
    permuted_accuracies_mllm_qwen_72B(i) = 100*mean(behav_predict==permuted_label');
end
% chance level
chance_level_mllm_qwen_72B = mean(permuted_accuracies_mllm_qwen_72B);
% chance level 95% CI
lower_bound_mllm_qwen_72B = prctile(permuted_accuracies_mllm_qwen_72B, 2.5);
upper_bound_mllm_qwen_72B = prctile(permuted_accuracies_mllm_qwen_72B, 97.5);
fprintf('Chance Level MLLM Qwen 72B: %.4f\n', chance_level_mllm_qwen_72B);
fprintf('Approximate 95%% CI for Chance Level: [%.4f, %.4f]\n', lower_bound_mllm_qwen_72B, upper_bound_mllm_qwen_72B);


% get prediction for each object
for i_obj = 1:355
    behav_predict_obj_mllm_qwen_72B(i_obj,1) = 100*mean(behav_predict(any(triplet_testdata==i_obj,2))==1);
    % this below gives us the predictability of each object on average
    % (i.e. how difficult it is expected to predict choices with it irrespective of other objects)
    behav_predict_obj_prob_mllm_qwen_72B(i_obj,1) = 100*mean(behav_predict_prob(any(triplet_testdata==i_obj,2)));
end
% get 95% CI for this value across objects
behav_predict_acc_ci95_mllm_qwen_72B = 1.96*std(behav_predict_obj_mllm_qwen_72B)/sqrt(355);

%%%%%%%%%%%%%%%%%%%%%
% Get noise ceiling %
%%%%%%%%%%%%%%%%%%%%%
data_mllm = load('data/MLLM_qwen_72B/qwen_72B_VL_noise_ceiling_results.mat');
probabilities_mllm = data_mllm.probabilities;
noise_ceiling_mllm_qwen_72B = mean(probabilities_mllm) * 100;
noise_ceiling_std_mllm_qwen_72B = std(probabilities_mllm) * 100;
noise_ceiling_ci95_mllm_qwen_72B = 1.96 * noise_ceiling_std_mllm_qwen_72B / sqrt(length(probabilities_mllm));


%% LLM_qwen_7B
% load embedding
spose_embedding30= load(fullfile('data/LLM_qwen_7B/qwen_7B_spose_embedding_sorted_final.txt'));
% get dot product (i.e. proximity)
dot_product30= spose_embedding30*spose_embedding30';

% load 10% validation (i.e., test) data
triplet_testdata = load(fullfile('data/LLM_qwen_7B/test_triplets_llm.txt'))+1; % 0 index -> 1 index
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate how much variance can be explained in the test set %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
behav_predict = zeros(length(triplet_testdata),1);
behav_predict_prob = zeros(length(triplet_testdata),1);
rng(42) % for reproducibility
for i = 1:length(triplet_testdata)
    sim(1) = dot_product30(triplet_testdata(i,1),triplet_testdata(i,2));
    sim(2) = dot_product30(triplet_testdata(i,1),triplet_testdata(i,3));
    sim(3) = dot_product30(triplet_testdata(i,2),triplet_testdata(i,3));
    [m,mi] = max(sim); % people are expected to choose the pair with the largest dot product
    if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
    if mi == 1
        behav_predict(i,1) = 3;
    elseif mi == 2
        behav_predict(i,1) = 2;
    else
        behav_predict(i,1) = 1;
    end
    behav_predict_prob(i,1) = exp(sim(mi))/sum(exp(sim)); % get choice probability
end
% get overall prediction (predict choice == 1)
behav_predict_acc_llm_qwen_7B = 100*mean(behav_predict==3)

num_permutations = 1000;
permuted_accuracies_llm_qwen_7B = zeros(1, num_permutations);
for i = 1:num_permutations
    permuted_label = randi([1, 3], 1, length(behav_predict));
    permuted_accuracies_llm_qwen_7B(i) = 100*mean(behav_predict==permuted_label');
end
% chance level
chance_level_llm_qwen_7B = mean(permuted_accuracies_llm_qwen_7B);
% chance level 95% CI
lower_bound_llm_qwen_7B = prctile(permuted_accuracies_llm_qwen_7B, 2.5);
upper_bound_llm_qwen_7B = prctile(permuted_accuracies_llm_qwen_7B, 97.5);
fprintf('Chance Level LLM Qwen 7B: %.4f\n', chance_level_llm_qwen_7B);
fprintf('Approximate 95%% CI for Chance Level: [%.4f, %.4f]\n', lower_bound_llm_qwen_7B, upper_bound_llm_qwen_7B);


% get prediction for each object
for i_obj = 1:355
    behav_predict_obj_llm_qwen_7B(i_obj,1) = 100*mean(behav_predict(any(triplet_testdata==i_obj,2))==1);
    % this below gives us the predictability of each object on average
    % (i.e. how difficult it is expected to predict choices with it irrespective of other objects)
    behav_predict_obj_prob_llm_qwen_7B(i_obj,1) = 100*mean(behav_predict_prob(any(triplet_testdata==i_obj,2)));
end
% get 95% CI for this value across objects
behav_predict_acc_ci95_llm_qwen_7B = 1.96*std(behav_predict_obj_llm_qwen_7B)/sqrt(355);

%%%%%%%%%%%%%%%%%%%%%
% Get noise ceiling %
%%%%%%%%%%%%%%%%%%%%%
data_llm = load('data/LLM_qwen_7B/qwen_7B_noise_ceiling_results.mat');
probabilities_llm = data_llm.probabilities;
noise_ceiling_llm_qwen_7B = mean(probabilities_llm) * 100;
noise_ceiling_std_llm_qwen_7B = std(probabilities_llm) * 100;
noise_ceiling_ci95_llm_qwen_7B = 1.96 * noise_ceiling_std_llm_qwen_7B / sqrt(length(probabilities_llm));



%% Human
spose_embedding30= load(fullfile('data/Human/human_odd_one_out_spose_embedding_sorted_final.txt'));
% get dot product (i.e. proximity)
dot_product30= spose_embedding30*spose_embedding30';

% load 10% validation (i.e., test) data
triplet_testdata = load(fullfile('data/Human/test_triplets_human.txt'))+1; % 0 index -> 1 index

behav_predict = zeros(length(triplet_testdata),1);
behav_predict_prob = zeros(length(triplet_testdata),1);
rng(42) % for reproducibility
for i = 1:length(triplet_testdata)
    sim(1) = dot_product30(triplet_testdata(i,1),triplet_testdata(i,2));
    sim(2) = dot_product30(triplet_testdata(i,1),triplet_testdata(i,3));
    sim(3) = dot_product30(triplet_testdata(i,2),triplet_testdata(i,3));
    [m,mi] = max(sim); % people are expected to choose the pair with the largest dot product
    if sum(sim==m)>1, tmp = find(sim==m); mi = tmp(randi(sum(sim==m))); m = sim(mi); end % break ties choosing randomly (reproducible by use of rng)
    behav_predict(i,1) = mi;
    behav_predict_prob(i,1) = exp(sim(mi))/sum(exp(sim)); % get choice probability
end
% get overall prediction (predict choice == 1)
behav_predict_acc_human = 100*mean(behav_predict==1)

num_permutations = 1000;
permuted_accuracies_human = zeros(1, num_permutations);
for i = 1:num_permutations
    permuted_label = randi([1, 3], 1, length(behav_predict));
    permuted_accuracies_human(i) = 100*mean(behav_predict==permuted_label');
end
% chance level
chance_level_human = mean(permuted_accuracies_human);
% chance level 95% CI
lower_bound_human = prctile(permuted_accuracies_human, 2.5);
upper_bound_human = prctile(permuted_accuracies_human, 97.5);
fprintf('Chance Level Human: %.4f\n', chance_level_human);
fprintf('Approximate 95%% CI for Chance Level: [%.4f, %.4f]\n', lower_bound_human, upper_bound_human);

n_objects_human = size(spose_embedding30, 1);

behav_predict_obj_human = zeros(n_objects_human, 1);
for i_obj = 1:n_objects_human
    behav_predict_obj_human(i_obj,1) = 100*mean(behav_predict(any(triplet_testdata==i_obj,2))==1);
end
% get 95% CI for this value across objects
behav_predict_acc_ci95_human = 1.96*std(behav_predict_obj_human)/sqrt(n_objects_human);

%%%%%%%%%%%%%%%%%%%%%
% Get noise ceiling %
%%%%%%%%%%%%%%%%%%%%%
noise_ceiling_human = 57.32;
noise_ceiling_ci95_human = 0.1; 


%%%%%%%%%%%%%%%%
% Plot results %
%%%%%%%%%%%%%%%%
dosave = 1;

bar_x = [4, 10, 16, 22, 28];
bar_w = 4;

fig = figure('Position', [800 800 450 500], 'color', 'none');
hold on;

customColor1 = [.2 .6 .4];   % LLM_qwen_7B
customColor2 = [.6 .2 .4];   % MLLM_qwen_72B
customColor3 = [.2 .3 .8];   % Human
customColor4 = [.9 .6 .1];   % MLLM_qwen_7B
customColor5 = [.2 .3 .4];   % LLM_deepseek

%% ===================== noise ceiling =====================
% Human
patch([bar_x(1)-bar_w/2, bar_x(1)+bar_w/2, bar_x(1)+bar_w/2, bar_x(1)-bar_w/2], ...
      [noise_ceiling_human+noise_ceiling_ci95_human, noise_ceiling_human+noise_ceiling_ci95_human, ...
       noise_ceiling_human-noise_ceiling_ci95_human, noise_ceiling_human-noise_ceiling_ci95_human], ...
      [0.7 0.7 0.7], 'EdgeColor', 'none');
% LLM DeepSeek
patch([bar_x(2)-bar_w/2, bar_x(2)+bar_w/2, bar_x(2)+bar_w/2, bar_x(2)-bar_w/2], ...
      [noise_ceiling_llm_deepseek+noise_ceiling_ci95_llm_deepseek, noise_ceiling_llm_deepseek+noise_ceiling_ci95_llm_deepseek, ...
       noise_ceiling_llm_deepseek-noise_ceiling_ci95_llm_deepseek, noise_ceiling_llm_deepseek-noise_ceiling_ci95_llm_deepseek], ...
      [0.7 0.7 0.7], 'EdgeColor', 'none');
% LLM Qwen 7B
patch([bar_x(3)-bar_w/2, bar_x(3)+bar_w/2, bar_x(3)+bar_w/2, bar_x(3)-bar_w/2], ...
      [noise_ceiling_llm_qwen_7B+noise_ceiling_ci95_llm_qwen_7B, noise_ceiling_llm_qwen_7B+noise_ceiling_ci95_llm_qwen_7B, ...
       noise_ceiling_llm_qwen_7B-noise_ceiling_ci95_llm_qwen_7B, noise_ceiling_llm_qwen_7B-noise_ceiling_ci95_llm_qwen_7B], ...
      [0.7 0.7 0.7], 'EdgeColor', 'none');
% MLLM Qwen 7B
patch([bar_x(4)-bar_w/2, bar_x(4)+bar_w/2, bar_x(4)+bar_w/2, bar_x(4)-bar_w/2], ...
      [noise_ceiling_mllm_qwen_7B+noise_ceiling_ci95_mllm_qwen_7B, noise_ceiling_mllm_qwen_7B+noise_ceiling_ci95_mllm_qwen_7B, ...
       noise_ceiling_mllm_qwen_7B-noise_ceiling_ci95_mllm_qwen_7B, noise_ceiling_mllm_qwen_7B-noise_ceiling_ci95_mllm_qwen_7B], ...
      [0.7 0.7 0.7], 'EdgeColor', 'none');
% MLLM Qwen 72B
patch([bar_x(5)-bar_w/2, bar_x(5)+bar_w/2, bar_x(5)+bar_w/2, bar_x(5)-bar_w/2], ...
      [noise_ceiling_mllm_qwen_72B+noise_ceiling_ci95_mllm_qwen_72B, noise_ceiling_mllm_qwen_72B+noise_ceiling_ci95_mllm_qwen_72B, ...
       noise_ceiling_mllm_qwen_72B-noise_ceiling_ci95_mllm_qwen_72B, noise_ceiling_mllm_qwen_72B-noise_ceiling_ci95_mllm_qwen_72B], ...
      [0.7 0.7 0.7], 'EdgeColor', 'none');

%% ===================== Drawing =====================
bar(bar_x(1), behav_predict_acc_human,         'FaceColor', customColor3, 'EdgeColor', 'none', 'BarWidth', bar_w);
bar(bar_x(2), behav_predict_acc_llm_deepseek,  'FaceColor', customColor5, 'EdgeColor', 'none', 'BarWidth', bar_w);
bar(bar_x(3), behav_predict_acc_llm_qwen_7B,   'FaceColor', customColor1, 'EdgeColor', 'none', 'BarWidth', bar_w);
bar(bar_x(4), behav_predict_acc_mllm_qwen_7B,  'FaceColor', customColor4, 'EdgeColor', 'none', 'BarWidth', bar_w);
bar(bar_x(5), behav_predict_acc_mllm_qwen_72B, 'FaceColor', customColor2, 'EdgeColor', 'none', 'BarWidth', bar_w);

%% ===================== Error Bar =====================
errorbar(bar_x(1), behav_predict_acc_human,         behav_predict_acc_ci95_human,         'Color', [0 0 0], 'LineWidth', 3);
errorbar(bar_x(2), behav_predict_acc_llm_deepseek,  behav_predict_acc_ci95_llm_deepseek,  'Color', [0 0 0], 'LineWidth', 3);
errorbar(bar_x(3), behav_predict_acc_llm_qwen_7B,   behav_predict_acc_ci95_llm_qwen_7B,   'Color', [0 0 0], 'LineWidth', 3);
errorbar(bar_x(4), behav_predict_acc_mllm_qwen_7B,  behav_predict_acc_ci95_mllm_qwen_7B,  'Color', [0 0 0], 'LineWidth', 3);
errorbar(bar_x(5), behav_predict_acc_mllm_qwen_72B, behav_predict_acc_ci95_mllm_qwen_72B, 'Color', [0 0 0], 'LineWidth', 3);

%% ===================== Chance level =====================
for b = 1:5
    switch b
        case 1, cl = chance_level_human;         lb = lower_bound_human;         ub = upper_bound_human;
        case 2, cl = chance_level_llm_deepseek;  lb = lower_bound_llm_deepseek;  ub = upper_bound_llm_deepseek;
        case 3, cl = chance_level_llm_qwen_7B;   lb = lower_bound_llm_qwen_7B;   ub = upper_bound_llm_qwen_7B;
        case 4, cl = chance_level_mllm_qwen_7B;  lb = lower_bound_mllm_qwen_7B;  ub = upper_bound_mllm_qwen_7B;
        case 5, cl = chance_level_mllm_qwen_72B; lb = lower_bound_mllm_qwen_72B; ub = upper_bound_mllm_qwen_72B;
    end
    plot([bar_x(b)-bar_w/2, bar_x(b)+bar_w/2], [cl cl], '-r', 'LineWidth', 2);
    errorbar(bar_x(b), cl, (ub-lb)/2, 'Color', [0 0 0], 'LineWidth', 1.5);
end

%% ===================== Text =====================
txt_y_human         = (behav_predict_acc_human + 30) / 2;
txt_y_llm_deepseek  = (behav_predict_acc_llm_deepseek + 30) / 2;
txt_y_llm_qwen_7B   = (behav_predict_acc_llm_qwen_7B + 30) / 2;
txt_y_mllm_qwen_7B  = (behav_predict_acc_mllm_qwen_7B + 30) / 2;
txt_y_mllm_qwen_72B = (behav_predict_acc_mllm_qwen_72B + 30) / 2;

text(bar_x(1), txt_y_human,         'Human',          'Rotation', 90, 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 17);
text(bar_x(2), txt_y_llm_deepseek,  'DeepSeek-R1',    'Rotation', 90, 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 17);
text(bar_x(3), txt_y_llm_qwen_7B,   'Qwen2.5-7B',     'Rotation', 90, 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 17);
text(bar_x(4), txt_y_mllm_qwen_7B,  'Qwen2.5-VL-7B',  'Rotation', 90, 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 17);
text(bar_x(5), txt_y_mllm_qwen_72B, 'Qwen2.5-VL-72B', 'Rotation', 90, 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 17);

text(mean(bar_x), 69.3, 'noise ceiling', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'FontSize', 15);
text(mean(bar_x), 35, 'chance', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'red', 'FontSize', 15);

%% ===================== Axes =====================
xlim([0, 32]);
ylim([30, 80]);
ylabel('Odd-one-out accuracy (%)', 'FontSize', 16);
set(gca, 'Ytick', 30:5:80);

hax = gca;
set(gca, 'FontSize', 16);
hax.TickDir = 'both';
hax.XTick = [];
hax.XColor = [0 0 0];
hax.YColor = [0 0 0];
hax.LineWidth = 1.5;
hax.Box = 'off';

if dosave
    exportgraphics(fig, 'figures/noisecelling_5models.pdf', 'ContentType', 'vector');
end

%% ===================== Summary =====================
fprintf('\n============ Odd-One-Out Summary ============\n');
fprintf('  Human:          Acc=%.2f%% [CI95 +/-%.2f], NC=%.2f%%\n', behav_predict_acc_human, behav_predict_acc_ci95_human, noise_ceiling_human);
fprintf('  LLM DeepSeek:   Acc=%.2f%% [CI95 +/-%.2f], NC=%.2f%%\n', behav_predict_acc_llm_deepseek, behav_predict_acc_ci95_llm_deepseek, noise_ceiling_llm_deepseek);
fprintf('  LLM Qwen-7B:    Acc=%.2f%% [CI95 +/-%.2f], NC=%.2f%%\n', behav_predict_acc_llm_qwen_7B, behav_predict_acc_ci95_llm_qwen_7B, noise_ceiling_llm_qwen_7B);
fprintf('  MLLM Qwen-7B:   Acc=%.2f%% [CI95 +/-%.2f], NC=%.2f%%\n', behav_predict_acc_mllm_qwen_7B, behav_predict_acc_ci95_mllm_qwen_7B, noise_ceiling_mllm_qwen_7B);
fprintf('  MLLM Qwen-72B:  Acc=%.2f%% [CI95 +/-%.2f], NC=%.2f%%\n', behav_predict_acc_mllm_qwen_72B, behav_predict_acc_ci95_mllm_qwen_72B, noise_ceiling_mllm_qwen_72B);
fprintf('=============================================\n');