function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add column of ones to matrix X and set to a1.
a1 = [ones(m,1) X];

% Initialize z2 to the product of activation matrix a1 and Theta1.
z2 = a1*Theta1';

% Initialize a2 matrix to sigmoid of z2 with bias feature.
a2 = [ones(size(z2),1) sigmoid(z2)];

% Initialize z3 to the product of activation matrix a2 and Theta2.
z3 = a2*Theta2';

% Initialize a3 matrix to sigmoid of z3.
a3 = sigmoid(z3);

% Create matrix containing max probability of each row with respective index.
[max_probabilities, indeces_of_maxP] = max(a3, [], 2);

% Setting p to column vector containing class indeces of max probabilites
p = indeces_of_maxP;


% =========================================================================


end
