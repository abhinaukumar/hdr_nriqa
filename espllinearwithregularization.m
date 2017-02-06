%espldataset_X_ext = dataset_X_ext;
%[esplpcacoeff4,esplscores4,espllatents4] = princomp(espldataset_X_ext);
%contr = cumsum(espllatents4)./sum(espllatents4);
%contributions = contr(500)

%espl_trans_dataset_X_ext4 = espldataset_X_ext*esplpcacoeff4(:,1:500);
%espl_trans_dataset_X_ext4 = espl_trans_dataset_X_ext4(ind,:);
%espl_trans_dataset_X_ext4 = espl_trans_dataset_X_ext4';
%esplval_lcc4 = zeros(1,10);
%esplval_srocc4= zeros(1,10);
%espl_dataset3 = espl_trans_dataset_X_ext4';
%espl_y = combinedtargets(ind);
%shuffles = randperm(747);
%trans_largedataset = largedataset*largepcacoeff(:,1:400);

lambda = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 10, 30, 50, 100, 500, 1000]; 

%D = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000];
D = [400];
d = 1;

%K = 10;
K = 5;

train_lcc = zeros(10, length(lambda));
val_lcc = zeros(10, length(lambda));
tiny_lcc = zeros(10, length(lambda));

train_srocc = zeros(10, length(lambda));
val_srocc = zeros(10, length(lambda));
tiny_srocc = zeros(10, length(lambda));

train_lcc_inner = zeros(K,1);
val_lcc_inner = zeros(K,1);
tiny_lcc_inner = zeros(K,1);

train_srocc_inner = zeros(K,1);
val_srocc_inner = zeros(K,1);
tiny_srocc_inner = zeros(K,1);

%for d = 1:length(D) 
    trans_largedataset = largedataset*largepcacoeff(:,1:D(d));
for k = 1:20
    shuffles = randperm(747);
    for j = 1:length(lambda);
        for i = 1:K
            disp(i);

            %net12.performParam.regularization = lambda(j)/(1+lambda(j));
            %net12 = init(net12);


            shuffles = shuffles([ceil(747*(K-1)/K + 1):747 1:ceil(747*(K-1)/K)]);
            train_ind = shuffles(1:ceil(747*(K-1)/K));
            val_ind = shuffles(ceil(747*(K-1)/K + 1):end);

            train_set = trans_largedataset([7*train_ind 7*train_ind-1 7*train_ind-2 7*train_ind-3 7*train_ind-4 7*train_ind-5 7*train_ind-6],:);
            train_targets = largetargets([7*train_ind 7*train_ind-1 7*train_ind-2 7*train_ind-3 7*train_ind-4 7*train_ind-5 7*train_ind-6]);

            %[net12 tr12] = train(net12,train_set',train_targets');
            svrmodel = fitrlinear(train_set,train_targets,'Regularization','ridge','Lambda',lambda(j));

            val_nn_pred = predict(svrmodel,espllive(val_ind,:)*largepcacoeff(:,1:D(d)));
            temp = corrcoef(val_nn_pred,espltargets(val_ind));
            val_lcc_inner(i) = temp(2,1);
            val_srocc_inner(i) = spear(val_nn_pred,espltargets(val_ind));
            
            train_nn_pred = predict(svrmodel,espllive(train_ind,:)*largepcacoeff(:,1:D(d)));
            temp = corrcoef(train_nn_pred,espltargets(train_ind));
            train_lcc_inner(i) = temp(2,1);
            train_srocc_inner(i) = spear(train_nn_pred,espltargets(train_ind));
            
            tiny_pred = predict(svrmodel,(tinydataset*largepcacoeff(:,1:D(d))));
            temp = corrcoef(tiny_pred,tinytargets);
            tiny_lcc_inner(i) = temp(2,1);
            tiny_srocc_inner(i) = spear(tiny_pred,tinytargets);
            
        end
        val_lcc(k,j) = median(val_lcc_inner);
        val_srocc(k,j) = median(val_srocc_inner);
        train_lcc(k,j) = median(train_lcc_inner);
        train_srocc(k,j) = median(train_srocc_inner);
        tiny_lcc(k,j) = median(tiny_lcc_inner);
        tiny_srocc(k,j) = median(tiny_srocc_inner);
    end
end

val_lcc_med = median(val_lcc,1);
train_lcc_med = median(train_lcc,1);
tiny_lcc_med = median(tiny_lcc,1);

val_srocc_med = median(val_srocc,1);
train_srocc_med = median(train_srocc,1);
tiny_srocc_med = median(tiny_srocc,1);

best_val_lcc_dim(d) = max(val_lcc_med);
best_train_lcc_dim(d) = max(train_lcc_med);
best_tiny_lcc_dim(d) = max(tiny_lcc_med);

best_val_srocc_dim(d) = max(val_srocc_med);
best_train_srocc_dim(d) = max(train_srocc_med);
best_tiny_srocc_dim(d) = max(tiny_srocc_med);

%end

%{
close all;
plot(lambda,train_lcc_med,'b');
hold on;
plot(lambda,val_lcc_med,'g');
hold on;
plot(lambda,tiny_lcc_med,'y');
%}

close all;

plot(D,best_train_lcc_dim,'b');
hold on;
plot(D,best_val_lcc_dim,'g');
hold on;
plot(D,best_tiny_lcc_dim,'y');
hold on;

legend('Median Training LCC','Median Validation LCC','Median LCC obtained on testing on our dataset');

figure;
plot(D,best_train_srocc_dim,'b');
hold on;
plot(D,best_val_srocc_dim,'g');
hold on;
plot(D,best_tiny_srocc_dim,'y');
hold on;

legend('Median Training SROCC','Median Validation SROCC','Median SROCC obtained on testing on our dataset');