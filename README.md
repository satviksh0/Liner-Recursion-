#Implement linear regression from scratch

This code implements linear regression using gradient descent from scratch in Python. It begins by reading data from a CSV file using pandas. Two key functions are defined: loss_fn calculates the mean squared error loss for a given set of parameters (slope and intercept) and data points, and gradient_descent computes the gradients of the loss function with respect to the parameters and updates them using gradient descent.
The main loop iterates over a fixed number of epochs, performing gradient descent to update the slope (m) and intercept (b) parameters. The learning rate (L) and number of epochs (epochs) are configurable hyperparameters. Finally, the code visualizes the fitted line on a scatter plot of the data points.
This implementation offers insight into how linear regression works under the hood, using basic mathematical operations to iteratively optimize parameters and minimize the error between predicted and actual values.
