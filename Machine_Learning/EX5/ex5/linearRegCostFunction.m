function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h_theta=X*theta;
J1=sum((h_theta-y).^2)/(2*m);
J2=lambda*(sum(theta.^2)-(theta(1,1)^2))/(2*m);
J = J1+J2;

grad1=X'*(h_theta-y)/m;
% we shouldn't regularize theta 0
theta(1,1)=0;
grad2=lambda*theta/m;
grad=grad1+grad2;










% =========================================================================

grad = grad(:);

end
