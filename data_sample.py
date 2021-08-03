import pickle

#pickle_in = open("./data_3.pickle","rb")

#pickle_in = open("./cifar10_train_10_data.pickle","rb")

pickle_in = open("./cifar10_color_train_10_data.pickle","rb")

data = pickle.load(pickle_in)

data = data[0:30]

#for i in data:
    #print(i)
    #print()
    #print()