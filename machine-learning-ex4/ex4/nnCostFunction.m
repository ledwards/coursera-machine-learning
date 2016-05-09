function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta_1 and Theta_2, the weight matrices
% for our 2 layer neural network
Theta_1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta_2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta_1_grad = zeros(size(Theta_1));
Theta_2_grad = zeros(size(Theta_2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

K = num_labels;
X = [ones(m,1) X];

for i = 1:m
  a_1 = X(i,:);
  z_2 = a_1 * Theta_1';
  a_2 = sigmoid(z_2);

  a_2 = [1 a_2];
  z_3 = a_2 * Theta_2';
  a_3 = sigmoid(z_3);

  h = a_3;

  temp = y(i);
  y_i = zeros(1, K);
  y_i(temp) = 1;

  J = J + sum(-y_i .* log(h) - (1 - y_i) .* log(1 - h));
end;

J = 1 / m * J;

% Add regularization term

J = J + (lambda / (2 * m) * (sum(sumsq(Theta_1(:, 2:input_layer_size + 1))) + sum(sumsq(Theta_2(: ,2:hidden_layer_size + 1)))));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta_1_grad and Theta_2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta_1 and Theta_2 in Theta_1_grad and
%         Theta_2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

D_1 = zeros(size(Theta_1));
D_2 = zeros(size(Theta_2));

for t = 1:m
  a_1 = X(t, :);
  z_2 = a_1 * Theta_1';
  a_2 = [1 sigmoid(z_2)];
  z_3 = a_2 * Theta_2';
  a_3 = sigmoid(z_3);
  y_i = zeros(1, K);
  y_i(y(t)) = 1;

  delta_3 = a_3 - y_i;
  delta_2 = delta_3 * Theta_2 .* sigmoidGradient([1 z_2]);

  D_1 = D_1 + delta_2(2:end)' * a_1;
  D_2 = D_2 + delta_3' * a_2;
end;

Theta_1_grad = (1 / m) * D_1;
Theta_2_grad = (1 / m) * D_2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta_1_grad
%               and Theta_2_grad from Part 2.
%

Theta_1_grad(:, 2:input_layer_size + 1) = Theta_1_grad(:, 2:input_layer_size + 1) + lambda / m * Theta_1(:, 2:input_layer_size + 1);
Theta_2_grad(:, 2:hidden_layer_size + 1) = Theta_2_grad(:, 2:hidden_layer_size + 1) + lambda / m * Theta_2(:, 2:hidden_layer_size + 1);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta_1_grad(:) ; Theta_2_grad(:)];


end
