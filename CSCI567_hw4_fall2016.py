import hw_utils
from datetime import datetime
X_train, Y_train, X_test, Y_test = hw_utils.loaddata("MiniBooNE_PID.txt")

X_train_norm , X_test_norm = hw_utils.normalize(X_train, X_test)

# X_train_small = X_train[0:1000]
# Y_train = Y_train[0:1000]
#
# X_train_norm_small = X_train_norm[0:1000]
# X_test_norm_small = X_test_norm[0:100]
#
# X_test_small = X_test[0:100]
# Y_test_small = Y_test[0:100]

# # Part d linear

# print "Linear arch 1"
# linear_arch1 = [[50, 2],[50, 50, 2],[50, 50, 50, 2],[50, 50, 50, 50, 2]]
# start = datetime.now()
# hw_utils.testmodels(X_train_norm, Y_train, X_test_norm, Y_test,
#                     linear_arch1, 'linear', 'softmax',[0.0], 30, 1000, 0.001, [0.0], [0.0], False, False, 0)
# end = datetime.now()
# print "Time taken in Linear activations part 1 : " + str((end - start).total_seconds())

# print "\n\nLinear arch 2"
# linear_arch2 = [[50, 50, 2], [50, 500, 2], [50, 500, 300, 2], [50, 800, 500, 300, 2], [50, 800, 800, 500, 300, 2]]
# start = datetime.now()
# hw_utils.testmodels(X_train_norm, Y_train, X_test_norm, Y_test,
#                     linear_arch2, 'linear', 'softmax',[0.0], 30, 1000, 0.001, [0.0], [0.0], False, False, 0)
# end = datetime.now()
# print "Time taken in Linear activations part 2 : " + str((end - start).total_seconds())

# arch = [[50, 50, 2], [50, 500, 2], [50, 500, 300, 2], [50, 800, 500, 300, 2], [50, 800, 800, 500, 300, 2]]

# # Part e sigmoid
# print "\n\n Sigmoid"
# start = datetime.now()
# hw_utils.testmodels(X_train_norm, Y_train, X_test_norm, Y_test,
#                     arch, 'sigmoid', 'softmax',[0.0], 30, 1000, 0.001, [0.0], [0.0], False, False, 0)
# end = datetime.now()
# print "Time taken in Sigmoid activation : " + str((end - start).total_seconds())

# # Part f relu
# print "\n\n Relu"
# start = datetime.now()
# hw_utils.testmodels(X_train_norm, Y_train, X_test_norm, Y_test,
#                     arch, 'relu', 'softmax',[0.0], 30, 1000, 0.0005, [0.0], [0.0], False, False, 0)
# end = datetime.now()
# print "Time taken in ReLu activation : " + str((end - start).total_seconds())

arch = [[50, 800, 500, 300, 2]]
# L2 = [0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001]

# # Part g L2
# print "\n\n L2"
# start = datetime.now()
# hw_utils.testmodels(X_train_norm, Y_train, X_test_norm, Y_test,
#                     arch, 'relu', 'softmax', L2, 30, 1000, 0.0005, [0.0], [0.0], False, False, 0)
# end = datetime.now()
# print "Time taken in L2 regularization : " + str((end - start).total_seconds())

# # Part h early stopping and L2
# print "\n\n early stopping and L2"
# start = datetime.now()
# hw_utils.testmodels(X_train_norm, Y_train, X_test_norm, Y_test,
#                     arch, 'relu', 'softmax', L2, 30, 1000, 0.0005, [0.0], [0.0], False, True, 0)
# end = datetime.now()
# print "Time taken in Early stopping and L2 regularization : " + str((end - start).total_seconds())

# Part i Decay
# print "\n\n decay"
# L2 = [0.0000005]
# decay = [0.00001, 0.00005, 0.0001, 0.0003, 0.0007, 0.001]
# start = datetime.now()
# hw_utils.testmodels(X_train_norm, Y_train, X_test_norm, Y_test,
#                     arch, 'relu', 'softmax', L2, 100, 1000, 0.00001, decay, [0.0], False, False, 0)
# end = datetime.now()
# print "Time taken in SGD with weight decay : " + str((end - start).total_seconds())

# Part j Momentum
print "\n\n momentum"
L2 = [0.0]
decay = [0.00001] # best decay from i
momentum = [0.99, 0.98, 0.95, 0.9, 0.85]
start = datetime.now()
hw_utils.testmodels(X_train_norm, Y_train, X_test_norm, Y_test,
                    arch, 'relu', 'softmax', L2, 50, 1000, 0.00001, decay, momentum, True, False, 0)
end = datetime.now()
print "Time taken in momentum : " + str((end - start).total_seconds())

# # Part k combination
# print "\n\n combination"
# L2 = [0.0] # Best value from h
# decay = [0.0] # best decay from i
# momentum = [0.0] # best value from j
# start = datetime.now()
# hw_utils.testmodels(X_train_norm, Y_train, X_test_norm, Y_test,
#                     arch, 'relu', 'softmax', L2, 100, 1000, 0.00001, decay, momentum, True, True, 0)
# end = datetime.now()
# print "Time taken in combining the above : " + str((end - start).total_seconds())

# # Part l grid search
# print "\n\n grid search"
# arch = [[50, 50, 2], [50, 500, 2], [50, 500, 300, 2], [50, 800, 500, 300, 2], [50, 800, 800, 500, 300, 2]]
# L2 = [0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001]
# decay = [0.00001, 0.0005, 0.0001]
# momentum = [0.99]
# start = datetime.now()
# hw_utils.testmodels(X_train_norm, Y_train, X_test_norm, Y_test,
#                     arch, 'relu', 'softmax', L2, 100, 1000, 0.00001, decay, momentum, True, True, 0)
# end = datetime.now()
# print "Time taken in Grid search with cross validation : " + str((end - start).total_seconds())