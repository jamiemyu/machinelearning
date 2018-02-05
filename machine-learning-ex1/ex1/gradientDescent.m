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
    %
    sumAmount1 = 0;
    sumAmount2 = 0;
    for i = 1:m
        h = theta(1) + theta(2) * X(i,2);
        sumAmount1 = sumAmount1 + ((h - y(i)) * X(i,1));
        sumAmount2 = sumAmount2 + ((h - y(i)) * X(i,2));
    end

    theta(1) = theta(1) - alpha * (1 / m) * sumAmount1;
    theta(2) = theta(2) - alpha * (1 / m) * sumAmount2;
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    fprintf('The computed cost is %f\n', J_history(iter));
end

end
