import hw_utils

X_train, Y_train, X_test, Y_test = hw_utils.loaddata("MiniBooNE_PID.txt")

X_train_norm , X_test_norm = hw_utils.normalize(X_train, X_test)

X_train_small = X_train[0:1000]
Y_train_small = Y_train[0:1000]

X_train_norm_small = X_train_norm[0:1000]
X_test_norm_small = X_test_norm[0:100]

X_test_small = X_test[0:100]
Y_test_small = Y_test[0:100]

# Part d linear
linear_arch1 = [[50, 2],[50, 50, 2],[50, 50, 50, 2],[50, 50, 50, 50, 2]]
hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    linear_arch1, 'linear', 'softmax',[0.0], 30, 1000, 0.001, [0.0], [0.0], False, False, 0)

linear_arch2 = [[50, 50, 2], [50, 500, 2], [50, 500, 300, 2], [50, 800, 500, 300, 2], [50, 800, 800, 500, 300, 2]]

hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    linear_arch2, 'linear', 'softmax',[0.0], 30, 1000, 0.001, [0.0], [0.0], False, False, 0)

arch = [[50, 50, 2], [50, 500, 2], [50, 500, 300, 2], [50, 800, 500, 300, 2], [50, 800, 800, 500, 300, 2]]

# Part e sigmoid
hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    arch, 'sigmoid', 'softmax',[0.0], 30, 1000, 0.001, [0.0], [0.0], False, False, 0)

# Part f relu
hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    arch, 'relu', 'softmax',[0.0], 30, 1000, 0.0005, [0.0], [0.0], False, False, 0)

arch = [[50, 800, 500, 300, 2]]
L2 = [0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001]

# Part g L2
hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    arch, 'relu', 'softmax', L2, 30, 1000, 0.0005, [0.0], [0.0], False, False, 0)

# Part h early stopping and L2
hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    arch, 'relu', 'softmax', L2, 30, 1000, 0.0005, [0.0], [0.0], False, True, 0)

# Part i Decay
L2 = [0.0000005]
decay = [0.00001, 0.00005, 0.0001, 0.0003, 0.0007, 0.001]
hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    arch, 'relu', 'softmax', L2, 100, 1000, 0.00001, decay, [0.0], False, False, 0)

# Part j Momentum
L2 = [0.0]
decay = [0.0] # best decay from i
momentum = [0.99, 0.98, 0.95, 0.9, 0.85]
hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    arch, 'relu', 'softmax', L2, 50, 1000, 0.00001, decay, momentum, True, False, 0)

# Part k combination
L2 = [0.0] # Best value from h
decay = [0.0] # best decay from i
momentum = [0.0] # best value from j
hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    arch, 'relu', 'softmax', L2, 100, 1000, 0.00001, decay, momentum, True, True, 0)

# Part l grid search
arch = [[50, 50, 2], [50, 500, 2], [50, 500, 300, 2], [50, 800, 500, 300, 2], [50, 800, 800, 500, 300, 2]]
L2 = [0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001]
decay = [0.00001, 0.0005, 0.0001]
momentum = [0.99]
hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    arch, 'relu', 'softmax', L2, 100, 1000, 0.00001, decay, momentum, True, True, 0)