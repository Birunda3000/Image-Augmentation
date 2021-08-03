import pickle

pickle_in = open("./Dados_para_testes/cifar10_color_train_10_data.pickle","rb")

data_cifar10_10 = pickle.load(pickle_in)

pickle_in = open("./Dados_para_testes/mnist_train_10_data.pickle","rb")

data_mnist_11 = pickle.load(pickle_in)