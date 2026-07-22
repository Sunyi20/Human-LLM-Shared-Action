clear all; close all; clc;
base_dir = pwd;
addpath(base_dir)
addpath(genpath(fullfile('../helper_functions')))

h = fopen(fullfile('colors.txt')); % get list of colors in hexadecimal format
col = zeros(0,3);
while 1
    l = fgetl(h);
    if l == -1, break, end
    
    col(end+1,:) = reshape(sscanf(l(2:end).','%2x'),3,[]).'/255; % hex2rgb
    
end
fclose(h);

col(1,:) = [];
col([1 2 3],:) = col([2 3 1],:);

colors = col([1 20 3 38 9 7 62 57 13 6 24 25 50 48 36 53 46 28 62 18 15 58 2 11 40 45 27 55 36 30 34 31 41 16 27 61 17 36 57 25 63],:); colors(end+1:49,:) = col(8:56-length(colors),:);
colors(46,:) = colors(46,:)-0.2; 

colors = colors([1 2 3 4 6 5 12 8 10 9 13 11 7 15 18 14 16 19 21 17 22 33 17 23 20 27 26 19 24 37 20 28 47 31 39 30 36 43 29 35 38 9 6 25 49 40 42 37 44 25 41 12 20 45 7 41 46 2 23 34 5 33 13 31 40 32],:);

colors([20 28 30 31 41 42 43 45 50 52 53 55 56 58 59 61 62 63 64 65],:) = 1/255*...
    [[146 78 167];
    [143 141 58];
    [255 109 246];
    [71 145 205];
    [0 118 133];
    [204 186 45];
    [0 222 0];
    [222 222 0];
    [100 100 100];
    [40 40 40];
    [126 39 119];
    [177 177 0];
    [50 50 150];
    [120 120 50];
    [250 150 30];
    [40 40 40];
    [220 220 220];
    [90 170 220];
    [140 205 150];
    [40 170 225]];

clear col h l


embedding = load(fullfile('data/MLLM_qwen_7B/qwen_7B_VL_spose_embedding_sorted_final.txt'));

fprintf('Embedding size: %d x %d\n', size(embedding, 1), size(embedding, 2));

spose_sim = embedding2sim(embedding);
dissim = 1 - spose_sim;

rng(42);
[Y2, stress] = mdscale(dissim, 2, 'criterion', 'metricstress');
fprintf('MDS stress: %.4f\n', stress);

embedding_scaled = embedding;
embedding_scaled(:,1) = embedding_scaled(:,1) * 0.1;
[~, clustid] = max(embedding_scaled, [], 2);

rng(1);
perplexity1 = 5;    
perplexity2 = 30;

D = dissim / max(dissim(:));
P = 1/2 * (d2p(D, perplexity1, 1e-5) + d2p(D, perplexity2, 1e-5));

figure('Visible', 'off');
Ytsne = tsne_p(P, clustid, Y2);
close();

Ytsne_centered = Ytsne - mean(Ytsne, 1);
Ytsne_normalized = Ytsne_centered / max(abs(Ytsne_centered(:)));

save('v2/spose_embedding_sorted_merge_tsne_MLLM_qwen_7B.mat', 'Ytsne');
save('v2/spose_embedding_sorted_merge_tsne_MLLM_qwen_7B_normalized.mat', 'Ytsne_normalized');

param_sets = {
    2.5, 20, 'low_perp';    
    5,   30, 'mid_perp';   
    10,  50, 'high_perp'; 
};

for p = 1:size(param_sets, 1)
    rng(1);
    p1 = param_sets{p, 1};
    p2 = param_sets{p, 2};
    tag = param_sets{p, 3};
    
    P_test = 1/2 * (d2p(D, p1, 1e-5) + d2p(D, p2, 1e-5));
    figure('Visible', 'off');
    Y_test = tsne_p(P_test, clustid, Y2);
    close();

    Ytsne_test = Y_test;
    save(sprintf('v2/tsne_%s.mat', tag), 'Ytsne_test');
end