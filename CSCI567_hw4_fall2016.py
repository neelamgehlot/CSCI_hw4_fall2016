import hw_utils

X_train, Y_train, X_test, Y_test = hw_utils.loaddata("MiniBooNE_PID.txt")

X_train_norm , X_test_norm = hw_utils.normalize(X_train, X_test)

X_train_small = X_train[0:1000]
Y_train_small = Y_train[0:1000]

X_train_norm_small = X_train_norm[0:1000]

print X_train_small.shape
print Y_train_small.shape
print X_train_norm_small.shape