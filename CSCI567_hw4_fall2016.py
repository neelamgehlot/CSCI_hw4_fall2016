import hw_utils

X_train, Y_train, X_test, Y_test = hw_utils.loaddata("MiniBooNE_PID.txt")

X_train_normalized , X_test_normalized = hw_utils.normalize(X_train, X_test)

print X_train.shape
print Y_train.shape
print X_test.shape
print Y_test.shape