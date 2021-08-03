# import sys
import threading
# from threading import Thread, Lock
# import time
# import random

import matplotlib.pyplot as plt


def print_list_img(lista_de_imagens):

    print('Numero imagens na base - {}'.format(len(lista_de_imagens)))

    plt.figure(figsize=(10,10))
    m = 100
    if( len(lista_de_imagens) < 100 ):
        m = len(lista_de_imagens)
    for i in range(m):
        plt.subplot(10,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(lista_de_imagens[i][0])#, cmap='gray')
        plt.xlabel("{}\n({})".format( lista_de_imagens[i][1], '-' ), color='white')    
    plt.show()

class augmentor(threading.Thread):
    def __init__(self, augmentor_id: int, dados: list, times: int, pipe):
        self.meu_id = augmentor_id
        self.dados = dados
        self.times = times#numero de imagens geradas por imagens dada
        self.pipe = pipe
        self.result = []
        # Atenção essa instrução deve ser chamada para iniciar a thread
        threading.Thread.__init__(self)

    def run(self):
        if True: #self.meu_id == 0: #or self.meu_id == 1:

            for i in range(len(self.dados)):

                self.result.append(self.dados[i])
                
                self.result += self.pipe.operar(self.dados[i][0], self.dados[i][1], self.times)

            '''for i in range(len(self.result)):
                print(self.result[i])
                print()
                print('-----------------------------------------------------')
                print()'''
            print('Thread ID - {}'.format(self.meu_id))
            print_list_img(self.result)

def dividir_base(base, n):  # n = numero de itens por thread
    for i in range(0, len(base), n):
        yield base[i:i + n]

def call_thread(img_per_thread, data, pipe_instance, image_per_image):
    threads = []
    data_array = list(dividir_base(data, img_per_thread))
    numbers_of_threads = int(len(data) / img_per_thread)


    for i in range(numbers_of_threads):
        new_augmentor = augmentor(augmentor_id = i, dados = data_array[i], times = image_per_image, pipe = pipe_instance)
        threads.append(new_augmentor)
        new_augmentor.start()
    for t in threads:
        t.join()
