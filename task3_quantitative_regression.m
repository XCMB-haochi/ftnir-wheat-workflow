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
cutoffs = [5 1000 100 60];
for i = 1:numel(toxins)
    rows = find(labels == i);
    y = data.nominal_ug_kg(rows);
    [~, ~, strata] = histcounts(y, max(3, min(5, floor(numel(y)/10))));
    series = toxins(i);
    x = spectra(rows, :);
    tr = splitRows(strata, 0.7);
    yTrain = y(tr);
    yTest = y(~tr);
    folds = cvpartition(numel(yTrain), 'KFold', 5);
    for model = ["PLSR", "SVR", "BPNN"]
        [predictor, parameters] = fitPipeline(x(tr, :), yTrain, model, folds);
        prediction = predictor(x(~tr, :));
        metrics = struct();
        calibration = predictor(x(tr, :));
        error = prediction - yTest;
        metrics.RC2 = determination(yTrain, calibration);
        metrics.RP2 = determination(yTest, prediction);
        metrics.RMSEC = sqrt(mean((calibration - yTrain).^2));
        metrics.RMSEP = sqrt(mean(error.^2));
        metrics.RPD = std(yTest) / metrics.RMSEP;
        metrics.bias = mean(error);
        line = polyfit(yTest, prediction, 1);
        metrics.slope = line(1);
        metrics.intercept = line(2);
        metrics.RER = range(yTest) / metrics.RMSEP;
        above = yTest > cutoffs(i);
        flagged = prediction > cutoffs(i);
        metrics.falsePositive = sum(flagged & ~above);
        metrics.falseNegative = sum(~flagged & above);
        metrics.sensitivity = mean(flagged(above));
        metrics.specificity = mean(~flagged(~above));
        loss = error.^2;
        groups = findgroups(data.toxin(rows(~tr)), data.sample_id(rows(~tr)));
        sums = accumarray(groups, loss);
        counts = accumarray(groups, 1);
        stream = RandStream('mt19937ar', 'Seed', 122);
        draw = randi(stream, numel(counts), numel(counts), 10000);
        boot = sum(sums(draw), 1) ./ sum(counts(draw), 1);
        boot = sqrt(boot);
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
    scales = [(.002:.001:.005)' (.002:.002:.008)'];
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
    if model == "PLSR"
        grid = (1:min([30, size(X, 2)-1, floor(numel(Y)*.7)-1]))';
    else
        grid = [];
    end
    objective = @(par) validationLoss(X, Y, model, folds, par);
    if ~isempty(grid)
        scores = zeros(size(grid, 1), 1);
        for j = 1:numel(scores)
            scores(j) = objective(grid(j, :));
        end
        [~, j] = min(scores);
        p = grid(j, :);
    else
        vars = [optimizableVariable('C',[.01 1000],'Transform','log'), ...
            optimizableVariable('scale',[.01 100],'Transform','log')];
        vars(3) = optimizableVariable('epsilon',[.001 100],'Transform','log');
        opt = bayesopt(@(row) objective(table2array(row)), vars, ...
            'MaxObjectiveEvaluations', 40, 'Verbose', 0, 'PlotFcn', {}, ...
            'IsObjectiveDeterministic', true, 'AcquisitionFunctionName', 'expected-improvement-plus');
        p = table2array(opt.XAtMinObjective);
    end
    [X, Y] = augment(X, Y);
    fn = baseModel(X, Y, model, p);
end

function value = validationLoss(X, Y, model, folds, p)
    loss = zeros(size(Y));
    for k = 1:folds.NumTestSets
        tr = training(folds, k);
        va = test(folds, k);
        x = X(tr, :);
        y = Y(tr);
        state = rng;
        rng(2100+k);
        [x, y] = augment(x, y);
        rng(state);
        fn = baseModel(x, y, model, p);
        prediction = fn(X(va, :));
        loss(va) = (prediction - Y(va)).^2;
    end
    value = mean(loss);
end

function fn = baseModel(X, Y, model, p)
    if model == "PLSR"
        [~, ~, ~, ~, beta] = plsregress(X, Y, p(1));
        fn = @(v) [ones(size(v,1),1), v] * beta;
    else
        mdl = fitrsvm(X, Y, 'KernelFunction','gaussian','BoxConstraint',p(1), ...
            'KernelScale',p(2),'Epsilon',p(3));
        fn = @(v) predict(mdl, v);
    end
end

function r2 = determination(observed, predicted)
    total = sum((observed - mean(observed)).^2);
    if total == 0
        r2 = NaN;
    else
        r2 = 1 - sum((observed - predicted).^2) / total;
    end
end

function [fn, settings] = fitNetwork(X, Y)
    widths = [
        256 128 64
        300 150 75
        200 100 50
        350 175 88
        180 90 45
        240 120 60
        280 140 70
        320 160 80
        220 110 55
        260 130 65
        400 200 100
        160 80 40
    ];
    mask = false(size(Y));
    mask(randperm(numel(Y), round(.8*numel(Y)))) = true;
    [x, y] = augment(X(mask, :), Y(mask));
    count = size(x, 1);
    x = [x; X(~mask, :)];
    y = [y; Y(~mask)];
    best = Inf;
    for j = 1:30
        width = widths(1 + mod(j-1, size(widths, 1)), :);
        net = newNetwork(width, 2000);
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = 1:count;
        net.divideParam.valInd = count+1:size(x, 1);
        net.divideParam.testInd = [];
        [~, record] = train(net, x', y');
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
    net = train(net, x', y');
    fn = @(v) net(v')';
    settings = struct('width', selected, 'epochs', stop);
end

function net = newNetwork(width, epochs)
    net = fitnet(width, 'trainscg');
    net.performFcn = 'mse';
    net.trainParam.epochs = epochs;
    net.trainParam.max_fail = 150;
    net.trainParam.goal = 1e-10;
    net.trainParam.showWindow = false;
end
