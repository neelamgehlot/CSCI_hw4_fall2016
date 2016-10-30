import hw_utils

X_train, Y_train, X_test, Y_test = hw_utils.loaddata("MiniBooNE_PID.txt")

X_train_norm , X_test_norm = hw_utils.normalize(X_train, X_test)

X_train_small = X_train[0:1000]
Y_train_small = Y_train[0:1000]

X_train_norm_small = X_train_norm[0:1000]
X_test_norm_small = X_test_norm[0:100]

X_test_small = X_test[0:100]
Y_test_small = Y_test[0:100]

print "start"

hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    [50, 2], 'linear', 'softmax',[0.0], 30, 1000, 0.001, 0.0, 0.0, False, False, 1)

hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    [50, 50, 2], 'linear', 'softmax',[0.0], 30, 1000, 0.001, 0.0, 0.0, False, False, 1)

hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    [50, 50, 50, 2], 'linear', 'softmax',[0.0], 30, 1000, 0.001, 0.0, 0.0, False, False, 1)

hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    [50, 50, 50, 50, 2], 'linear', 'softmax',[0.0], 30, 1000, 0.001, 0.0, 0.0, False, False, 1)

hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    [50, 50, 2], 'linear', 'softmax',[0.0], 30, 1000, 0.001, 0.0, 0.0, False, False, 1)

hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    [50, 500, 2], 'linear', 'softmax',[0.0], 30, 1000, 0.001, 0.0, 0.0, False, False, 1)

hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    [50, 500, 300, 2], 'linear', 'softmax',[0.0], 30, 1000, 0.001, 0.0, 0.0, False, False, 1)

hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    [50, 800, 500, 300, 2], 'linear', 'softmax',[0.0], 30, 1000, 0.001, 0.0, 0.0, False, False, 1)

hw_utils.testmodels(X_train_small, Y_train_small, X_test_small, Y_test_small,
                    [50, 800, 800, 500, 300, 2], 'linear', 'softmax',[0.0], 30, 1000, 0.001, 0.0, 0.0, False, False, 1)

print "end"