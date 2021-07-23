import pickle

pickle_in = open("./data_3.pickle","rb")
data = pickle.load(pickle_in)

data = data[0:10]

for i in data:
    print(i)
    print()
    print()