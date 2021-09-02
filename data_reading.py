import pickle
import random

#pickle_in = open("./Dados_para_testes/cifar10_color_train_10_data.pickle","rb")
pickle_in = open("../bases/pickle/imagens/cifar10/cifar10-train-pickle.pickle","rb")

data_cifar10_10 = pickle.load(pickle_in)

random.shuffle(data_cifar10_10)

data_cifar10_10 = data_cifar10_10[0:30] 


#pickle_in = open("./Dados_para_testes/mnist_train_10_data.pickle","rb")
pickle_in = open("../bases/pickle/imagens/mnist/mnist-train-pickle.pickle","rb")

data_mnist_11 = pickle.load(pickle_in)

random.shuffle(data_mnist_11)

data_mnist_11 = data_mnist_11[0:30]