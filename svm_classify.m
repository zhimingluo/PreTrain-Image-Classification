function accuracy = svm_classify(database, nRounds, tr_num, c, useFlip)

WholeFea = zeros(4096, length(database.fea_path));
for ii = 1:length(database.fea_path)
    fea = load(database.fea_path{ii});
    WholeFea(:,ii) = fea.feature;
end
% L2-Normalization
WholeFea = bsxfun(@times, WholeFea, 1./max(1e-5,sqrt(sum(WholeFea.^2,1))));


if useFlip
    WholeFea_flip = zeros(4096, length(database.fea_flip_path));
    for ii = 1:length(database.fea_flip_path)
        feapath = database.fea_flip_path{ii};
        fea = load(feapath);
        WholeFea_flip(:,ii) = fea.feature;
    end
    %L2-Normalization
    WholeFea_flip = bsxfun(@times, WholeFea_flip, 1./max(1e-5,sqrt(sum(WholeFea_flip.^2,1)))); 
end


clabel = unique(database.label);
nclass = length(clabel);
accuracy = zeros(nRounds, 1);

for ii = 1:nRounds,
    fprintf('Round: %d...\n', ii);
    tr_idx = [];
    ts_idx = [];
    
    for jj = 1:nclass,
        idx_label = find(database.label == clabel(jj));
        num = length(idx_label);     
        idx_rand = randperm(num);
        tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];   %index of training samples
        ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))]; %index of testing samples
    end
      
    % load the training features
    tr_fea = WholeFea(:,tr_idx);
    tr_label = database.label(tr_idx);
    
    if useFlip
        tr_fea = [tr_fea WholeFea_flip(:,tr_idx)];
        tr_label = [tr_label; tr_label];
    end
    
    options = ['-q -c ' num2str(c)];
    model = train(double(tr_label), sparse(tr_fea'), options);
    clear tr_fea tr_label;
    
    ts_fea = WholeFea(:,ts_idx);
    ts_label = database.label(ts_idx);
    
    % load the testing features
    [C] = predict(ts_label, sparse(ts_fea'), model);
    
    % normalize the classification accuracy by averaging over different classes
    acc = zeros(nclass, 1);
    
    for jj = 1 : nclass,
        c = clabel(jj);
        idx = find(ts_label == c);
        curr_pred_label = C(idx);
        curr_gnd_label = ts_label(idx);
        acc(jj) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
    end
    
    accuracy(ii) = mean(acc);
    fprintf('Classification accuracy for round %d: %.2f%%\n', ii, accuracy(ii)*100);
end