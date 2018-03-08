import numpy as np

x = np.arange(4).reshape(2,2)
w = np.arange(16).reshape(2,2,2,2)
print x
print '********'
y = x*w
print w
print '********'
print y
print '********'
VP = np.sum(np.sum(y,axis=3),axis=2)
print VP
print '********'
print w[np.where(VP > 40)]
w[np.where(VP > 40)] = 100
print w
VP[VP <= 40] = 0
VP[VP  > 40] = 1
