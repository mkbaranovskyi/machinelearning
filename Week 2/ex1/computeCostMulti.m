function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h = X * theta; % predictions h (m x 1) = design matrix X (m x n) x vector theta (n x 1)
sqrErrors = (h - y) .^ 2; % sqeErrors vector (m x 1) = h (m x 1) - y (m x 1)
J = 1 / (2 * m) * sum(sqrErrors);

% =========================================================================

end
