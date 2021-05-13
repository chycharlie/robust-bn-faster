% Compute the empirical parental configuration probabilities of a known-structure Bayes net.
% Input: the graph structure (parent) and the samples (X).
% Output: the empirical parental configuration probabilities (pi).
% That is, for an index k = (i,j), pi(k) = the empirical Pr[parent(X_i) = j].

function [pi] = empirical_parental_prob(parent, X)
    % N = number of samples, d = number of variables, m = size of conditional probability table.
    N = size(X, 1);
    d = size(parent, 1);
    deg = zeros(d, 1);
    % Compute m.
    m = 0;
    for i = 1:d
        deg(i) = numel(parent{i});
        m = m + 2^deg(i);
    end
    
    % Compute pi.
    pi = zeros(m, 1);
    k = 0;
    for i = 1:d
        for j = 1:2^deg(i)
            parent_config = dec2bin(j-1, deg(i)) - '0';
            % cnt01 = the number of times that [parent(i) = j] occurred in X.
            if (deg(i) == 0)
                cnt01 = N;
                parent_config = [];
            else
                cnt01 = sum(all(bsxfun(@eq, X(:, parent{i}), parent_config), 2));
            end
            k = k + 1;
            pi(k) = cnt01 / N;
        end
    end
end