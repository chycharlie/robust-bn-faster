% Experiments on robust learning of fixed-structure Bayes Nets (BN):
%   Y. Cheng, H. Lin.
%   Robust Learning of Fixed-Structure Bayesian Networks in Nearly-Linear Time.
%   Proceedings of the 9th International Conference on Learning Representations (ICLR), 2021.

% Compare the performance of MLE_noNoise, MLE, Filtering, and our algorithm (performance = error in total variation distance) for various d and m.
% d = number of variables, m = number of conditional probabilities, eps = fraction of corruption / precision parameter, N = number of samples = 10*m/eps^2.

clear
rng(37, 'twister');  % Fix random seed for replication/debug.

% Whether sparse matrices are used (fXq is N by m but each row is d-sparse).
% Sparse matrices save space but could make the computation slower.
use_sparse_matrix = 0;


%%% STEP 1: Construct the ground-truth BN (graph structure and conditional probabilities).
% Feel free to replace it with your favorite BN.

% Options for whether the dependence graph of the true BN is empty, a random tree, or a random graph.
bn_is_empty = 0;
bn_is_tree = 0;
fprintf('BN_is_empty = %d, BN_is_tree = %d\n', bn_is_empty, bn_is_tree);

% Target number of parameters and dimension of the true BN.
target_m = 500;
if (bn_is_tree)
    d = target_m / 2;
else
    d = 50;
end

% Generate the in-degree of every node.
parent = cell(d, 1);
if (bn_is_empty)
    deg = zeros(d, 1);
    m = d;
elseif (bn_is_tree)
    % Tree case: the degree is 0 for the root and 1 for other nodes.
    deg = [0; ones(d-1, 1)];
    m = d+d-1;
else
    % Graph case: start with the empty graph; increase the degree of a random node until m reaches the targer m.
    deg = zeros(d, 1);
    m = d;
    while (m < target_m)
        i = randi(d);
        if (deg(i) < i-1)
            m = m - 2^deg(i) + 2^(deg(i)+1);
            deg(i) = deg(i) + 1;
        end
    end
end

% Generate the graph structure (edges go from nodes with smaller index to ones with larger index).
% Generate the conditional probabilities p(i, a), drawn i.i.d from [0, 1/4] \cup [3/4, 1].
p = zeros(m, 1);
m = 0;
for i = 1:d
    parent{i} = randsample(i-1, deg(i))';
    for j = 1:2^deg(i)
        m = m + 1;
        if (rand() > 0.5)
            p(m) = rand(1)/4 + 3/4;
        else
            p(m) = rand(1)/4;
        end
    end
end
fprintf('d = %d, m = %d\n', d, m);


%%% STEP 2: Take (1-eps)*N samples from the ground-truth BN.
% We can allow an adaptive adversary. For simplicity, we use Huber's contamination model here.
eps = 0.1;
N = 10 * floor(m / eps^2);
X = zeros(round((1-eps)*N), d);
k = 0;
% This is much faster than looping over N.
for i = 1:d
    for j = 1:2^deg(i)
        % For every node i and every parental configuration j, select the samples in which parent[i] = j, draw X_i randomly.
        parent_config = dec2bin(j-1, deg(i)) - '0';
        k = k + 1;
        matched_rows = all(bsxfun(@eq, X(:, parent{i}), parent_config), 2);
        X(matched_rows, i) = rand(sum(matched_rows), 1) < p(k);
    end
end


%%% STEP 3: Evaluate MLE without noise (gold standard).
p_MLE_noNoise = empirical_cond_mean(parent, X);
fprintf('\td_TV(p, p_MLE_noNoise) = %f\n', dtv_bn(parent, p, p_MLE_noNoise));


%%% STEP 4: Take eps*N corrupted samples arbitrarily.
% We draw Y from another BN (with different structure) here. This decision is arbitrary.
% Feel free to replace Y with anything.
if (bn_is_empty || bn_is_tree)
    % Draw corrupted samples from a product distribution if the true BN is empty or is a tree.
    % The mean of the product distribution is i.i.d in [0, 1].
    p_noise = rand(1, d);
    Y = rand(round(eps*N), d);
    Y = bsxfun(@le, Y, p_noise);
else
    % Draw corrupted samples from a random tree BN if the true BN is a graph.
    % The conditional probabilities are drawn i.i.d from [0, 1/4] \cup [3/4, 1].
    Y = zeros(round(eps*N), d);
    k = 0;
    for i = 2:d
        parent_noise_i = randsample(i-1, min(i-1, 1))';
        for j = 1:2
            k = k + 1;
            if (rand() > 0.5)
                p_noise_k = rand(1)/4 + 3/4;
            else
                p_noise_k = rand(1)/4;
            end
            % Pi(i, j) == 1 if (y(parent_noise_i) == j-1).
            matched_rows = bsxfun(@eq, Y(:, parent_noise_i), j-1);
            Y(matched_rows, i) = rand(sum(matched_rows), 1) < p_noise_k;
        end
    end
end
X = [X; Y];
N = size(X, 1);


%%% STEP 5: Evaluate MLE (i.e., empirical conditional mean) as baseline #1.
p_MLE = empirical_cond_mean(parent, X);
fprintf('\td_TV(p, p_MLE) = %f\n', dtv_bn(parent, p, p_MLE));


%%% STEP 6: Evaluate Filtering as baseline #2:
%   Y. Cheng, I. Diakonikolas, D. Kane, A. Stewart.
%   Robust Learning of Fixed-Structure Bayesian Networks.
%   In Proceedings of the 32nd Conference on Neural Information Processing Systems (NeurIPS), 2018.

% We only run one iteration of Filtering, which is faster and gives reasonable results.

% Expand the sample dimension from d to m.  Fill in the unknown information with empirical conditional means.
% Compute and store f(X,q).
if (use_sparse_matrix == 0)
    fXq = zeros(N, m);
    k = 0;
    % Do not loop over N -- too slow.
    for i = 1:d
        for j = 1:2^deg(i)
            parent_config = dec2bin(j-1, deg(i)) - '0';
            k = k + 1;  % k = (i, j)
            if deg(i) == 0
                fXq(:, k) = X(:, i) - p_MLE(k);
            else
                matched_config = all(bsxfun(@eq, X(:, parent{i}), parent_config), 2);
                fXq(matched_config, k) = X(matched_config, i) - p_MLE(k);
                fXq(~matched_config, k) = 0;
            end
        end
    end
else
    % Too slow to access a sparse matrix repeatly by index, create it all at once.
    fXq_i = zeros(N*d, 1);
    fXq_j = zeros(N*d, 1);
    fXq_v = zeros(N*d, 1);
    fXq_nnz = 0;
    k = 0;
    for i = 1:d
        for j = 1:2^deg(i)
            parent_config = dec2bin(j-1, deg(i)) - '0';
            k = k + 1;
            if deg(i) == 0
                fXq_i(fXq_nnz+1:fXq_nnz+N) = (1:N)';
                fXq_j(fXq_nnz+1:fXq_nnz+N) = k;
                fXq_v(fXq_nnz+1:fXq_nnz+N) = X(:, i) - p_MLE(k);
                fXq_nnz = fXq_nnz + N;
            else
                matched_config = all(bsxfun(@eq, X(:, parent{i}), parent_config), 2);
                sum_matched = sum(matched_config);
                fXq_i(fXq_nnz+1:fXq_nnz+sum_matched) = find(matched_config);
                fXq_j(fXq_nnz+1:fXq_nnz+sum_matched) = k;
                fXq_v(fXq_nnz+1:fXq_nnz+sum_matched) = X(matched_config, i) - p_MLE(k);
                fXq_nnz = fXq_nnz + sum_matched;
            end
        end
    end
    fXq = sparse(fXq_i, fXq_j, fXq_v, N, m);
end

% Compute the top eigenvalue and eigenvector of off-diag(cov(Fxq - q)).
cov_Fxq_q = full(fXq' * fXq) / N;
cov_Fxq_q = cov_Fxq_q - diag(diag(cov_Fxq_q));
[v1, lambda1] = eigs(cov_Fxq_q, 1);
% Project the (expanded) samples along the direction v1.
projection_data_pair = [abs(fXq * v1) X];
% Sort by the absolute value of the projection (first column).
sorted_pair = sortrows(projection_data_pair);
% Remove eps-fraction of the sample farthest from the projected mean.
p_filter = empirical_cond_mean(parent, sorted_pair(1:round((1-eps)*N), 2:end));
fprintf('\td_TV(p, p_filter) = %f\n', dtv_bn(parent, p, p_filter));


%%% STEP 7: Evaluate our algorithm [Cheng, Lin, ICLR 2021].
pi_S = empirical_parental_prob(parent, X);
q_S = p_MLE;
% Avoid dividing by zero.
pi_S(pi_S == 0) = 1/N;
q_S(q_S == 0) = 1/N;
q_S(q_S == 1) = 1 - 1/N;

n_itr = floor(log(d)) + 1;
% Maintain a guess q for p and gradually improve q.
q = q_S;
for itr = 1:n_itr
    % Expand the domain size from d to m.  Fill in the unknown information with emperical mean.
    % Compute and store f(X,q) for the current guess q.
    if (use_sparse_matrix == 0)
        fXq = zeros(N, m);
        k = 0;
        for i = 1:d
            for j = 1:2^deg(i)
                parent_config = dec2bin(j-1, deg(i)) - '0';
                k = k + 1;  % k = (i, j)
                if deg(i) == 0
                    fXq(:, k) = X(:, i) - q(k);
                else
                    matched_config = all(bsxfun(@eq, X(:, parent{i}), parent_config), 2);
                    fXq(matched_config, k) = X(matched_config, i) - q(k);
                    fXq(~matched_config, k) = 0;
                end
            end
        end
    end
    if (use_sparse_matrix == 1)
        fXq_i = zeros(N*d, 1);
        fXq_j = zeros(N*d, 1);
        fXq_v = zeros(N*d, 1);
        fXq_nnz = 0;
        k = 0;
        for i = 1:d
            for j = 1:2^deg(i)
                parent_config = dec2bin(j-1, deg(i)) - '0';
                k = k + 1;
                if deg(i) == 0
                    fXq_i(fXq_nnz+1:fXq_nnz+N) = (1:N)';
                    fXq_j(fXq_nnz+1:fXq_nnz+N) = k;
                    fXq_v(fXq_nnz+1:fXq_nnz+N) = X(:, i) - q(k);
                    fXq_nnz = fXq_nnz + N;
                else
                    matched_config = all(bsxfun(@eq, X(:, parent{i}), parent_config), 2);
                    sum_matched = sum(matched_config);
                    fXq_i(fXq_nnz+1:fXq_nnz+sum_matched) = find(matched_config);
                    fXq_j(fXq_nnz+1:fXq_nnz+sum_matched) = k;
                    fXq_v(fXq_nnz+1:fXq_nnz+sum_matched) = X(matched_config, i) - q(k);
                    fXq_nnz = fXq_nnz + sum_matched;
                end
            end
        end
        fXq = sparse(fXq_i, fXq_j, fXq_v, N, m);
    end
    
    % Scale f(X,q) so that the covariance matrix of the good samples is close to I.
    hat_fXq = fXq ./ (ones(N, 1) * sqrt(pi_S .* q_S .* (1-q_S))');

    % Use any robust mean estimation algorithm that works under stability conditions of first and second moments.
    % We provide two examples of robust mean algorithms.  Choose one or implement your own.
    nu = robust_mean_filter(hat_fXq, eps);  % one-iteration filtering.
    % nu = robust_mean_pgd(hat_fXq, eps);  % projected gradient descent.
    
    % Update q using the result returned by the robust mean estimation oracle.
    q = max(0, min(1, q + nu .* sqrt(q_S .* (1-q_S) ./ pi_S)));
    
    % If you want to see how q is improving between iteratons.
    % fprintf('\t\titr = %d, d_TV(p, q) = %f\n', itr, dtv_bn(parent, p, q));
end
p_faster = q;
fprintf('\td_TV(p, p_faster) = %f\n', dtv_bn(parent, p, p_faster));