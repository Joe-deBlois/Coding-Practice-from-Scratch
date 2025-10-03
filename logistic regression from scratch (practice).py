import numpy as np
import math
import matplotlib.pyplot as plt

y_train1 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
X_train1 = [[5.5, 4, 3.8], 
           [5.2, 3, 2.01], 
           [6.0, 5, 1.4], 
           [5.5, 3, 2.5], 
           [5.4, 2, 3.1], 
           [5.7, 6, 1.4], 
           [5.8, 6, 1.2], 
           [5.3, 5, 3.6], 
           [5.4, 2, 2.9], 
           [5.9, 5, 1.01]]
y_test1 = [1, 1, 1, 0, 0]
X_test1 = [[5.5, 4, 3.8], 
           [5.2, 3, 2.01], 
           [6.0, 5, 1.4], 
           [5.5, 3, 2.5], 
           [5.4, 2, 3.1]]
y_train2 = [0, 0, 0, 1, 1, 1]
X_train2 = [[0.5, 1.5], 
    [1, 1],       
    [1.8, 2],   
    [3, 2.8],  
    [3.3, 3.5], 
    [4, 4]]

y_test2 = [0, 0, 0, 1, 1, 1]
X_test2 = [[1, 2],  
    [1.5, 1.8],
    [2, 1.5],
    [3, 3],   
    [3.5, 4], 
    [4, 3.5]]


####################################################
#                 Function Method                  #
####################################################
#region function method
def log_reg_fit(y_train, X_train, max_iter, tol, lr):
    
    #initalization
    num_features = len(X_train[0])
    num_samples = len(y_train)
    iter = 1
    w = [0] * num_samples
    b = 0

    while iter <= max_iter: 
        #step 1: raw z-values
        z = []
        for i in range(num_samples):
            z.append(0)
        for i in range(num_samples): 
            for j in range(num_features):
                z[i] += ((w[j] * X_train[i][j]))
            z[i] += b
        #print("z at iteration " +  str(iter) +  ":")
        #print(z)

        #step 2: sigmoid probabilities
        y_hat = []
        for z_i in z: 
            y_hat.append(1/(1+ math.e **(-z_i)))
        #print("y_hats at iteration " +  str(iter) +  ":")
        #print(y_hat)

        #step 3: compute errors 
        errors = []
        for i in range(num_samples):
            errors.append(y_train[i]-y_hat[i])
        #print("errors at iteration " +  str(iter) +  ":")
        #print(errors)

        #step 4: new gradients for weights
        weight_grads = []
        for j in range(num_features):
            weight_grads.append(0)
        for i in range(num_samples):
            for j in range(num_features):
                weight_grads[j] += (errors[i] * X_train[i][j])
        for j in range(num_features):
            weight_grads[j] = weight_grads[j] / num_samples
        #print("weight grads at iteration " +  str(iter) +  ":")
        #print(weight_grads)

        #step 5: new gradient for bias
        grad_bias = sum(errors)/num_samples
        #print("new bias grad at iteration " +  str(iter) +  ": " +  str(grad_bias))

        #step 6: update parameters with learning rate 
        w_new = []
        for j in range(num_features):
            w_new.append(0)
        for j in range(num_features):
            w_new[j] = w[j] + (lr * weight_grads[j])
        #print("new weights at end of iteration " + str(iter) + ":")
        #print(w_new)
    
        b_new = b + (lr * grad_bias)
        #print("new bias at end of iteration " + str(iter) + ": " + str(b_new))

        #step 7: stopping condition met? 
        if iter >= max_iter:
            return w_new, b_new
        #check if ALL weights and bias changed less than tol
        if all(abs(w_new[j] - w[j]) <= tol for j in range(num_features)) and abs(b_new - b) <= tol:
            return w_new, b_new
        else: 
            w = w_new
            b = b_new
            iter += 1

#Coefficients in log. regression are important for interpretation, AND the ranges per column are not very wide at all; So I will not standardize X

def log_reg_predict_and_evaluate(y_test, X_test, weights, bias):
    num_samples = len(y_test)
    num_features = len(X_test[0])

    #step 1: raw z-values
    z = [0]* num_samples
    for i in range(num_samples):
        for j in range(num_features):
            z[i] += ((weights[j] * X_test[i][j]))
        z[i] +=bias
    #print("predicted z values: ")
    #print(z)
    
    #step 2: sigmoid probabilities
    y_preds = [0] * num_samples
    for i in range(num_samples):
        y_preds[i] = (1/(1+ math.e **(-z[i])))
    for i in range(num_samples):
        if y_preds[i] <= 0.5:
            y_preds[i] = 0
        else:
            y_preds[i] = 1
    #print("predicted y values: ")
    #print(y_preds)

#region: plot predicted vs. actual points for each predictor
    for j in range(num_features):
    
        plotted_actual = False
        plotted_pred = False

        #plot actual & predicted values
        for i in range(num_samples):
            if not plotted_actual: 
                plt.scatter(X_test[i][j], y_test[i], marker = 'o', linestyle = 'solid', color = '#0023FF', alpha = 0.5, label = "Actual") #true values
                plotted_actual = True
            else:
                plt.scatter(X_test[i][j], y_test[i], marker = 'o', linestyle = 'solid', color = '#0023FF', alpha = 0.5) #true values
            if not plotted_pred:
                plt.scatter(X_test[i][j], y_preds[i], marker = 'o', linestyle ='solid', color = '#FF0000', alpha = 0.5, label = "Predicted") #predicted values
                plotted_pred = True
            else:
                 plt.scatter(X_test[i][j], y_preds[i], marker = 'o', linestyle ='solid', color = '#FF0000', alpha = 0.5)
        #Generate x-values for sigmoid curve
        #Generate 300 since we need a smooth line (denser points)
        X_test_nparray = np.array(X_test)
        x_vals = np.linspace(np.min(X_test_nparray[:, j]), np.max(X_test_nparray[:,j]), 300)

        #Simulate changing one feature while keeping all others constant at their mean
        mean_features = np.mean(X_test, axis = 0) #axis = 0 means "take the mean across rows, for each column/feature"
        X_input = np.tile(mean_features, (300,1))
        X_input[:,j] = x_vals

        #Compute sigmoid curve on these 300 points using w and b 
        sigmoid_curve = []
        #1: Compute raw z-values for X_input
        z = [0]* len(X_input)
        for i in range(len(X_input)):
            for k in range(num_features):
                z[i] += ((weights[k] * X_input[i][k]))
            z[i] +=bias

        #2: turn raw values into sigmoid probabilities
        for i in range(len(X_input)):
            sigmoid_curve.append(1/(1 + math.e ** (-z[i])))
    
        #plot sigmoid (probability) curve
        plt.plot(x_vals, sigmoid_curve, color='#2DA60F', linewidth=2, label=f"Sigmoid Curve for Feature {j} (others fixed at mean)")
        
        plt.title(f"Predicted vs. Actual (feature{j})")
        plt.xlabel(f"Feature {j}")
        plt.ylabel("Response")
        plt.legend()
        plt.show()
#endregion

    #step 3: compute errors 
    errors = [0] * num_samples
    for i in range(num_samples):
        if y_preds[i] != y_test[i]:
            errors[i] = 1
        else: 
            errors[i] = 0
    #print("Errors: ")
    #print(errors)
    return errors

#train model
log_reg_fit1 = log_reg_fit(y_train1, X_train1, max_iter = 200, tol = 1e-4, lr = 0.1)
print("Weights:", log_reg_fit1[0])
print("Bias: ", log_reg_fit1[1])
log_reg_fit2 = log_reg_fit(y_train2, X_train2, max_iter = 200, tol = 1e-4, lr = 0.1)

#test and evaluate
log_reg_predict_and_evaluate1 = log_reg_predict_and_evaluate(y_test1, X_test1, log_reg_fit1[0], log_reg_fit1[1])
log_reg_predict_and_evaluate2 = log_reg_predict_and_evaluate(y_test2, X_test2, log_reg_fit2[0], log_reg_fit2[1])

print(f"Learned weights for dataset 1: {log_reg_fit1[0]}\nLearned bias for dataset 1: {log_reg_fit1[1]} \nMisclassifications from dataset 1: {log_reg_predict_and_evaluate1}")

print(f"Learned weights for dataset 2: {log_reg_fit2[0]}\nLearned bias for dataset 2: {log_reg_fit2[1]} \n Misclassifications from dataset 2: {log_reg_predict_and_evaluate2}")

print("Where 0 = correct, 1 = misclassification")
#endregion

####################################################
#               Constructor Method                 #
####################################################
#region constructor method
class LogisticRegression():
   
    def __init__(self, y_train, y_test, X_train, X_test, max_iter, tol, lr):
        self.y_train = y_train
        self.y_test = y_test
        self.X_train = X_train
        self.X_test = X_test
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.w = None
        self.b = None
    
    def raw_z_vals(self, X, w, b):
        z = [0] * len(X)
        for i in range(len(X)): 
            for j in range(len(X[0])):
                z[i] += ((w[j] * X[i][j]))
            z[i] += b
        return z
    
    def sigmoid(self, z):
        y_hat = []
        for z_i in z: 
            y_hat.append(1/(1+ math.e **(-z_i)))
        return y_hat
    
    def compute_misclassifications(self, y_actuals, y_hats):
        errors = []
        for i in range(len(y_actuals)):
            errors.append(y_actuals[i]-y_hats[i])
        return errors

    def fit(self):
        #initalization
        num_samples = len(self.X_train)
        num_features = len(self.X_train[0])
        iter = 1
        w = [0] * num_features
        b = 0

        while iter <= self.max_iter: 
            #step 1: raw z-values
            z = self.raw_z_vals(self.X_train, w, b)

            #step 2: sigmoid probabilities
            y_hats = self.sigmoid(z)

            #step 3: compute errors 
            errors = self.compute_misclassifications(self.y_train, y_hats)

            #step 4: new gradients for weights
            weight_grads = [0] * num_features
            for i in range(num_samples):
                for j in range(num_features):
                    weight_grads[j] += (errors[i] * self.X_train[i][j])
            for j in range(num_features):
                weight_grads[j] = weight_grads[j] / num_samples

            #step 5: new gradient for bias
            grad_bias = sum(errors)/num_samples

            #step 6: update parameters with learning rate 
            w_new = [0] * num_features
            for j in range(num_features):
                w_new[j] = w[j] + (self.lr * weight_grads[j])
    
            b_new = b + (self.lr * grad_bias)
            
            #step 7: stopping condition met? 
            if iter >= self.max_iter:
                self.w = w_new
                self.b = b_new
            #check if ALL weights and bias changed less than tol
            if all(abs(w_new[j] - w[j]) <= self.tol for j in range(num_features)) and abs(b_new - b) <= self.tol:
                self.w = w_new
                self.b = b_new
            else: 
                w = w_new
                b = b_new
                iter += 1
    


logistic_model1 = LogisticRegression(y_train1, y_test1, X_train1, X_test1, max_iter = 200, tol = 1e-4, lr = 0.1)
logistic_model1.fit()
print("Weights: ", logistic_model1.w)
print("Bias: ", logistic_model1.b)




#endregion