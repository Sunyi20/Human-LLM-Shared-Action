clear; clc;

activities = load('data/varpartition/activities_rdm.mat');
activities_rdm = activities.rdm_vector;

categories = load('data/varpartition/categories_rdm.mat');
categories_rdm = categories.rdm_vector;

target = load('data/varpartition/target_rdm.mat');
target_rdm = target.rdm_vector;

model1 = activities_rdm;
model2 = categories_rdm;
model3 = target_rdm;

all_models = load('data/varpartition/all_LLM_rdm.mat');
model_names = fieldnames(all_models);

var_labels = {'三模型共享', '模型1+2共享', '模型2+3共享', '模型1+3共享', '模型1独特', '模型2独特', '模型3独特'};

for m = 1:length(model_names)
    model_name = model_names{m};
    if strcmpi(model_name, '__header__') || strcmpi(model_name, '__version__') || strcmpi(model_name, '__globals__')
        continue;
    end
    
    fprintf('\n======================================================\n');
    fprintf('Processing: %s\n', model_name);
    
    current_rdm_matrix = all_models.(model_name);
    if ndims(current_rdm_matrix) == 3
        num_subjects = size(current_rdm_matrix, 1);
        num_stimuli = size(current_rdm_matrix, 2);
        num_pairs = num_stimuli * (num_stimuli - 1) / 2;
        
        rdm_data = zeros(num_subjects, num_pairs);
        for i = 1:num_subjects
            single_rdm_matrix = squeeze(current_rdm_matrix(i, :, :));
            rdm_data(i, :) = squareform(single_rdm_matrix);
        end
    else
        temp = squareform(current_rdm_matrix);
        if size(temp, 1) > 1
             rdm_data = temp'; 
        else
             rdm_data = temp;
        end
    end

    
    tic;
    varpart_results = sim_varpart_nocv(rdm_data, model1, model2, model3);
    elapsed_time = toc;
    
    fprintf('Finish! Time: %.2f s\n', elapsed_time);

    %% Print Result
    for i = 1:7
        mean_var = varpart_results.pred(i);
        fprintf('%s: %.4f\n', var_labels{i}, mean_var);
    end

    fprintf('\nTotal Variance: %.4f\n', varpart_results.total);

    save_filename = sprintf('%s_varpart_results.mat', model_name);
    save(save_filename, 'varpart_results', 'rdm_data', 'model1', 'model2', 'model3');
end
