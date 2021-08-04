import threading
import math
import matplotlib.pyplot as plt

def print_list_img(lista_de_imagens, classes:list=None, imagens_por_linha:int=6, imagens_por_coluna:int=6):
    linha = imagens_por_linha
    coluna = imagens_por_coluna
    numero = math.ceil( len(lista_de_imagens)/ (linha * coluna) )
    print('Numero imagens - {}'.format(len(lista_de_imagens)))
    if classes != None:
        if classes == None:
            classes = range(len(lista_de_imagens))
        for j in range(numero):
            plt.figure(figsize=(10,10))
            m = (linha * coluna)    
            if( len(lista_de_imagens) < (linha * coluna) ):
                m = len(lista_de_imagens)
            for i in range(m):
                plt.subplot(linha, coluna, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(lista_de_imagens[i][0], cmap='gray')
                plt.ylabel("Digito - {}".format( lista_de_imagens[i][1]), color='white')
                plt.xlabel("{}".format( classes[lista_de_imagens[i][1]]), color='white')   
            plt.show()
            lista_de_imagens = lista_de_imagens[i:]
    else:
        for j in range(numero):
            plt.figure(figsize=(10,10))
            m = (linha * coluna)    
            if( len(lista_de_imagens) < (linha * coluna) ):
                m = len(lista_de_imagens)
            for i in range(m):
                plt.subplot(linha, coluna, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(lista_de_imagens[i], cmap='gray')  
            plt.show()
            lista_de_imagens = lista_de_imagens[i:]

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
        if True:#---------------------------------------------------------------------------retirar
            for i in range(len(self.dados)):
                self.result.append(self.dados[i])
                self.result += self.pipe.operar(self.dados[i][0], self.dados[i][1], self.times)
            print('Thread {}: Executada com sucesso'.format(self.meu_id))
            global result
            result += self.result

def dividir_base(base, n):  # n = numero de itens por thread
    for i in range(0, len(base), n):
        yield base[i:i + n]

result = []

def call_thread(img_per_thread, data, pipe_instance, image_per_image):
    
    threads = []
    data_array = list(dividir_base(data, img_per_thread))
    
    numbers_of_threads = int( math.ceil( len(data) / img_per_thread ) )

    for i in range(numbers_of_threads):
        new_augmentor = augmentor(augmentor_id = i, dados = data_array[i], times = image_per_image, pipe = pipe_instance)
        threads.append(new_augmentor)
        new_augmentor.start()
    for t in threads:
        t.join()
    

    
#trecho a ser analisado

#Original
    return result

    #x = result #quando se retorna result direto e se chama 2 vezes no codigo principal o retorno volta com o resultado das execuções anteriores 
    #result = []
    #return x