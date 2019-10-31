function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
   
    %{
    X     - (m x n) matrix
    h     - (m x 1) vector
    y     - (m x 1) vector
    theta - (n x 1) vector, since equal to features number X(1, :) - x0, x1, ... , xn
    Thus, we should (n x m) x (m x 1) to get (n x 1):  X' = (h - y)
    Or we can get theta as (1 x n) vector, so we'll do (1 x m) x (m x n): (h - y)' * X.
    Doesn't matter, just dimensions, the data is the same. Just remember: multiply 
    %}
    
    h = X * theta;
    difference = h - y; % difference between out prediction h and the actual value y, the less, the better
    theta -= (alpha / m) * (X' * difference); 

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
