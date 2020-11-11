#!/usr/bin/env python

#In this file, sklearn is used to create a support vector machine
#The vector machine will classify feature vectors to relevance


from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
print(clf.predict([[.5,.4]]))
