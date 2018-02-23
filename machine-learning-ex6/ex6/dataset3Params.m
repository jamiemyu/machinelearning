function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

params_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

sigmaError = 0;
sigmaIndex = 0;
cIndex = 0;
for i = 1:length(params_vec)
    for j = 1:length(params_vec)
        C = params_vec(i);
        sigma = params_vec(j);
        
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        predictions = svmPredict(model,Xval);

        currSigmaError = mean(double(predictions ~= yval));
        if (j == 1)
            sigmaError = currSigmaError;
            sigmaIndex = 1;
        else
            if (currSigmaError < sigmaError)
                sigmaError = currSigmaError;
                sigmaIndex = j;
            end
        end
    end
    
    if (i == 1)
        error = sigmaError;
        cIndex = 1;
    else
        if (sigmaError < error)
            error = sigmaError;
            cIndex = i;
        end
    end
end

C = params_vec(cIndex);
sigma = params_vec(sigmaIndex);

% =========================================================================

end
