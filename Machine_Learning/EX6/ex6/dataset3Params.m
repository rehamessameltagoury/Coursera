function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.3;
sigma = 0.1;
pred_error=0;

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

C_trial  = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_trial = C_trial;

% Number of trials for C and sigma
m = size(C_trial,2);

% Average number of wrong predictions, initialised to 1
missShot = 1;

for(i=1:m) 
    for(j=1:m) 
        % Get model from different C and sigma trial values
        model = svmTrain(X, y, C_trial(i), @(x1, x2) gaussianKernel(x1, x2, sigma_trial(j)));
        
        
        pred = svmPredict(model, Xval);
        
       
        if (missShot > mean(double(pred ~= yval))) 
            missShot = mean(double(pred ~= yval));
            C_temp = C_trial(i);
            sigma_temp = sigma_trial(j);
        end    
    end    
end

% Set the best C and sigma value to C and sigma
C = C_temp;
sigma = sigma_temp;


        








% =========================================================================

end
