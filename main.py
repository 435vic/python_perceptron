from numpy import array
import perceptron as per

tinputs = array([
    [0, 0],
    [0, 1],
    [1, 1],
])
toutputs = array([[0, 1, 1]]).T

p = per.Perceptron()
print("Predictions before training: ")
print(p.predict(tinputs))
p.train(tinputs, toutputs, 100000)
print("Predictions after training: ")
print(p.predict(tinputs))
print("Predict [1, 0] = ?")
print(p.predict(array([1, 0])))
