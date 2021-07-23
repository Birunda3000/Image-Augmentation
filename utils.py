# import sys
import threading
# from threading import Thread, Lock
# import time
# import random


class augmentor(threading.Thread):
    def __init__(self, augmentor_id: int, dados: list, times: int, pipe):
        self.meu_id = augmentor_id

        self.dados = dados
        # Atenção essa instrução deve ser chamada para iniciar a thread
        threading.Thread.__init__(self)

        self.times = times

        self.pipe = pipe

        self.result = []

    def run(self):
        if self.meu_id == 0: #or self.meu_id == 1:
            # self.result += self.dados
            for i in range(len(self.dados)):
                # print(self.dados[i])
                # print()
                # print()
                self.result += self.dados[i]
                self.result += self.pipe.operar(self.dados[i][0], self.dados[i][1], self.times)
            
            print(self.result[0])
            # print(len(self.dados))
            print(len(self.result))


def dividir_base(base, n):  # n = numero de itens por thread
    for i in range(0, len(base), n):
        yield base[i:i + n]

def call_thread(img_per_thread, data, pipe_instance):
    threads = []
    data_array = list(dividir_base(data, img_per_thread))
    numbers_of_threads = int(len(data) / img_per_thread)
    # for i in data:
    #     print(i)
    #     print()
    #     print()
    # for i in data_array:
    #     print(i)
    #     print()
    #     print()

    for i in range(numbers_of_threads):
        new_augmentor = augmentor(i, data_array[i], 3, pipe_instance)
        threads.append(new_augmentor)
        new_augmentor.start()
    for t in threads:
        t.join()
