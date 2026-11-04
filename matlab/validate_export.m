function report = validate_export(path)
%VALIDATE_EXPORT Independently check a Python-exported allocation in MATLAB.
% report = validate_export('runs/demo/validation.mat')
% No optimization toolbox is required. All AP/user indices are one-based.
d = load(path, 'net', 'solution');
n = d.net; s = d.solution;
M = double(n.M); K = double(n.K);
A = logical(s.A); z = logical(s.z(:)); x = double(s.x);
u = double(s.u); c = double(s.rC(:)); t = double(s.t);
assert(isequal(size(A), [M K]) && isequal(size(x), [M K]));
assert(isequal(size(u), [M K]) && numel(z) == K && numel(c) == K);
assert(all(isfinite([x(:); u(:); c; t])));
assert(all(ismember(d.solution.A(:), [0 1])) && all(ismember(d.solution.z(:), [0 1])));
p = x.^2; den = ones(K, 1);
for k = 1:K
    den(k) = 1 + sum(n.g(:, k, :) .* reshape(p, [M 1 K]), 'all');
end
rc = double(n.rho) * log2(1 + sum(n.a .* x, 1)'.^2 ./ den);
rc(~z) = 0; rn = zeros(M, K);
for k = 1:K
    if z(k), continue; end
    tail = 0;
    for pos = M:-1:1
        m = double(n.order(pos, k)); signal = n.a(m, k)^2 * p(m, k);
        if A(m, k)
            rn(m, k) = n.rho * log2(1 + signal / (den(k) + tail));
        end
        tail = tail + signal;
    end
end
fronthaulLoad = A(:, z) * c(z) + sum(u, 2); service = c + sum(u, 1)';
residuals = [0; -t; -x(:); -u(:); -c; sum(p, 2)-1; ...
    abs(x(~A)); abs(u(~A)); abs(c(~z)); reshape(abs(u(:, z)), [], 1); ...
    1-sum(A, 1)'; sum(A, 1)'-double(n.Dmax); ...
    (fronthaulLoad-n.C(:))./max(1, n.C(:)); ...
    (n.w(:)*t-service)./max(1, n.w(:)*t); ...
    (c-rc)./max(1, rc); reshape((u-rn)./max(1, rn), [], 1)];
report = struct('feasible', max(residuals) <= 2e-6, ...
    'maxResidual', max(residuals), 'service', service, 'fronthaulLoad', fronthaulLoad);
assert(report.feasible, 'Allocation violates the exported model');
fprintf('Feasible allocation, maximum residual %.3e\n', report.maxResidual);
end
