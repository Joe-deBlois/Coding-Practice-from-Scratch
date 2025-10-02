import numpy as np
import math

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

def log_reg_fit(y_train, X_train, max_iter, tol):
    
    #initalization
    iter = 1
    w = []
    for j in range(len(X_train[0])):
        w.append(0)
    b = 0
    lr = 0.1
    num_features = len(X_train[0])
    num_samples = len(y_train)

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
log_reg_fit1 = log_reg_fit(y_train1, X_train1, 200, 1e-4)
log_reg_fit2 = log_reg_fit(y_train2, X_train2, 200, 1e-4)

#test and evaluate
log_reg_predict_and_evaluate1 = log_reg_predict_and_evaluate(y_test1, X_test1, log_reg_fit1[0], log_reg_fit1[1])
log_reg_predict_and_evaluate2 = log_reg_predict_and_evaluate(y_test2, X_test2, log_reg_fit2[0], log_reg_fit2[1])

print(f"Learned weights for dataset 1: {log_reg_fit1[0]}\nLearned bias for dataset 1: {log_reg_fit1[1]} \nMisclassifications from dataset 1: {log_reg_predict_and_evaluate1}")

print(f"Learned weights for dataset 2: {log_reg_fit2[0]}\nLearned bias for dataset 2: {log_reg_fit2[1]} \n Misclassifications from dataset 2: {log_reg_predict_and_evaluate2}")

print("Where 0 = correct, 1 = misclassification")

####################################################
#               Constructor Method                 #
####################################################