if ~exist('inputFile', 'var') || strlength(string(inputFile)) == 0
    [file, folder] = uigetfile('*.csv', 'Select spectral data');
    if isequal(file, 0)
        return
    end
    inputFile = fullfile(folder, file);
end

rng(1717, 'twister');
data = readtable(inputFile, 'TextType', 'string');
names = string(data.Properties.VariableNames);
required = ["toxin", "sample_id", "replicate", "nominal_ug_kg"];
assert(all(ismember(required, names)), 'Missing input columns.');
columns = find(startsWith(names, "wn_"));
[wavenumbers, order] = sort(str2double(extractAfter(names(columns), "wn_")));
spectra = data{:, columns(order)};
assert(numel(wavenumbers) >= 11 && all(isfinite(wavenumbers)) && all(diff(wavenumbers) > 0));
assert(all(isfinite(spectra), 'all') && all(std(spectra, 0, 2) > 0));
spectra = sgolayfilt((spectra - mean(spectra, 2)) ./ std(spectra, 0, 2), 3, 11, [], 2);
toxins = ["AFB1", "DON", "T2", "ZEN"];
[valid, labels] = ismember(data.toxin, toxins);
assert(all(valid) && all(isfinite(data.nominal_ug_kg)) && ~any(ismissing(data.sample_id)));
assert(height(unique(data(:, {'toxin','sample_id','replicate'}), 'rows')) == height(data));
results = struct([]);
rows = (1:height(data))';
y = labels;
strata = y;
series = "All";
x = spectra(rows, :);
tr = splitRows(strata, 0.7);
yTrain = y(tr);
yTest = y(~tr);
folds = cvpartition(yTrain, 'KFold', 5);
for model = ["PLS-DA", "SVM", "KNN", "BPNN"]
    [predictor, parameters] = fitPipeline(x(tr, :), yTrain, model, folds);
    prediction = predictor(x(~tr, :));
    metrics = struct();
    M = confusionmat(yTest, prediction, 'Order', unique(y));
    metrics.confusion = M;
    metrics.accuracy = mean(prediction == yTest);
    metrics.precision = diag(M) ./ sum(M, 1)';
    metrics.recall = diag(M) ./ sum(M, 2);
    metrics.F1 = 2 * diag(M) ./ (sum(M, 1)' + sum(M, 2));
    loss = double(prediction == yTest);
    groups = findgroups(data.toxin(rows(~tr)), data.sample_id(rows(~tr)));
    sums = accumarray(groups, loss);
    counts = accumarray(groups, 1);
    stream = RandStream('mt19937ar', 'Seed', 122);
    draw = randi(stream, numel(counts), numel(counts), 10000);
    boot = sum(sums(draw), 1) ./ sum(counts(draw), 1);
    metrics.interval95 = prctile(boot, [2.5 97.5]);
    item = struct('series', series, 'model', model, 'parameters', parameters, ...
        'trainRows', rows(tr), 'testRows', rows(~tr), 'observed', yTest, ...
        'predicted', prediction, 'metrics', metrics);
    if isempty(results)
        results = item;
    else
        results(end+1) = item;
    end
end

function mask = splitRows(strata, fraction)
    mask = false(size(strata));
    for level = unique(strata(:))'
        idx = find(strata == level);
        n = max(1, min(numel(idx)-1, round(fraction*numel(idx))));
        mask(idx(randperm(numel(idx), n))) = true;
    end
end

function [x, y] = augment(X, Y)
    x = X;
    y = Y;
    scales = [.002 0; .004 0; .003 .006];
    for j = 1:size(scales, 1)
        noise = scales(j, 1) * randn(size(X));
        offset = scales(j, 2) * randn(size(X, 1), 1);
        x = [x; X + noise + offset];
        y = [y; Y];
    end
end

function [fn, p] = fitPipeline(X, Y, model, folds)
    if model == "BPNN"
        [fn, p] = fitNetwork(X, Y);
        return
    end
    if model == "PLS-DA"
        grid = (1:min([20, size(X, 2)-1, floor(numel(Y)*.7)-1]))';
    elseif model == "KNN"
        grid = [1; 3; 5; 7; 9; 11; 15; 21];
        grid = grid(grid <= floor(numel(Y)*.7));
    else
        [C, scale] = ndgrid([.1 1 10 100], [.001 .01 .1 1]);
        grid = [C(:), scale(:)];
    end
    objective = @(par) validationLoss(X, Y, model, folds, par);
    scores = zeros(size(grid, 1), 1);
    for j = 1:numel(scores)
        scores(j) = objective(grid(j, :));
    end
    [~, j] = min(scores);
    p = grid(j, :);
    fn = baseModel(X, Y, model, p);
end

function value = validationLoss(X, Y, model, folds, p)
    loss = zeros(size(Y));
    for k = 1:folds.NumTestSets
        tr = training(folds, k);
        va = test(folds, k);
        x = X(tr, :);
        y = Y(tr);
        fn = baseModel(x, y, model, p);
        prediction = fn(X(va, :));
        loss(va) = prediction ~= Y(va);
    end
    value = mean(loss);
end

function fn = baseModel(X, Y, model, p)
    if model == "PLS-DA"
        [~, ~, ~, ~, beta] = plsregress(X, double(Y == (1:max(Y))), p(1));
        fn = @(v) classifyScores([ones(size(v,1),1), v] * beta);
    elseif model == "KNN"
        mdl = fitcknn(X, Y, 'NumNeighbors', p(1));
        fn = @(v) predict(mdl, v);
    else
        learner = templateSVM('KernelFunction','gaussian','BoxConstraint',p(1),'KernelScale',p(2));
        mdl = fitcecoc(X, Y, 'Learners', learner, 'Coding', 'onevsall');
        fn = @(v) predict(mdl, v);
    end
end

function y = classifyScores(scores)
    [~, y] = max(scores, [], 2);
end

function [fn, settings] = fitNetwork(X, Y)
    widths = [
        512 256 128 64
        600 300 150 75
        700 350 175 88
        400 200 100 50
        450 225 112 56
        550 275 138 69
        384 192 96 48
        500 250 125 64
        640 320 160 80
    ];
    mask = splitRows(Y, .8);
    [x, y] = augment(X(mask, :), Y(mask));
    count = size(x, 1);
    x = [x; X(~mask, :)];
    y = [y; Y(~mask)];
    best = Inf;
    for j = 1:15
        width = widths(1 + mod(j-1, size(widths, 1)), :);
        net = newNetwork(width, 2500);
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = 1:count;
        net.divideParam.valInd = count+1:size(x, 1);
        net.divideParam.testInd = [];
        [~, record] = train(net, x', double(y == (1:max(Y)))');
        if record.best_vperf < best
            best = record.best_vperf;
            selected = width;
            stop = max(1, record.best_epoch);
        end
    end
    assert(isfinite(best), 'No finite validation loss.');
    [x, y] = augment(X, Y);
    net = newNetwork(selected, stop);
    net.divideFcn = 'dividetrain';
    net = train(net, x', double(y == (1:max(Y)))');
    fn = @(v) classifyScores(net(v')');
    settings = struct('width', selected, 'epochs', stop);
end

function net = newNetwork(width, epochs)
    net = patternnet(width, 'trainscg', 'crossentropy');
    net.trainParam.epochs = epochs;
    net.trainParam.max_fail = 180;
    net.trainParam.goal = 1e-10;
    net.trainParam.showWindow = false;
end
