# Image-Augmentation
## **--EM DESENVOLVIMENTO--**

Uma biblioteca para aumentar bases de dados de imagens
## Utilização
Para utilizar a biblioteca:

Primeiro se importa biblioteca
```python
import Aug
```
Declara-se o modelo
```python
pipe = Aug.Pipe()
```
Usa-se o metodo .add para adicionar as operações que serão utilizadas (em ordem)
```python
pipe.add(Aug.Random_Noise(0.7))
pipe.add(Aug.Random_Erasing(prob=1, rectangle_area=0.3, repetitions=2))
```
Passa a imagem e o numero de imagens a serem geradas como argumento da função operar, que retorna uma lista com as imagens geradas (como são atribuidas probabilidades para as operações é possivel que nenhuma seja aplicada, neste caso a imagem não sera adicionada podendo gerar um numero de imagens menor que o esperado)
```python
lista_de_imagens = []
lista_de_imagens = pipe.operar(image, 3)
```

## Operações
##### Toda operação vai receber um argumento " prob " que indica a probabilidade da operação ser aplicada ou não 
* **Rotação**
	* Descrição
		*  Roda a imagem um numero de graus aleatorio no intervalo especificado
	* Argumentos
		*  max_left_rotation:
			* Valor padrão: 90
		* max_right_rotation:
			* Valor padrão: 90
		* fill:
			* Valor padrão: 'edge'
	* Exemplo de chamada
```python
pipe.add(Aug.Rotacao(0.2, max_left_rotation=40, max_right_rotation=40))
```
		
* **Shift**
	* Descrição: 
	* Argumentos: 
 
* **Tilt**
	* Descrição: 
	* Argumentos: 

* **Skew**
	* Descrição:
	* Argumentos: (prob=unive, skew_type='RANDOM', magnitude=0.9) )

* **Invert**
	* Descrição: 
	* Argumentos: (prob=unive) )

* **Brilho**
	* Descrição: 
	* Argumentos: (prob=unive, min_factor= 0.1, max_factor= 10))

* **Color**
	* Descrição: 
	* Argumentos: (prob=unive, min_factor=-20, max_factor=20))

* **Contrast**
	* Descrição: 
	* Argumentos: (prob=unive, min_factor=-5, max_factor=5) )

* **Flip**
	* Descrição: 
	* Argumentos: (prob=unive))

* **Shear**
	* Descrição: 
	* Argumentos: (prob=unive, max_shear_left = 30, max_shear_right = 30 ))

* **Distort**
	* Descrição: 
	* Argumentos: (prob=unive, grid_width=4, grid_height=4, magnitude=8) )

* **Zoom**
	* Descrição: 
	* Argumentos: (prob=unive, min_factor=1, max_factor=9) )

* **Zoom_Random**
	* Descrição: 
	* Argumentos: (prob=unive, percentage_area=0.5, randomise=True))

* **Random_Erasing**
	* Descrição: 
	* Argumentos: (prob=unive, rectangle_area=0.3, repetitions=3))

* **Shift**
	* Descrição: 
	* Argumentos: (prob=unive, horizontal_max=0.8, vertical_max=0.8, randomise=True, fill='nearest') )

* **Rotacao**
	* Descrição: 
	* Argumentos: (prob=unive, max_left_rotation=90, max_right_rotation=90, fill='edge') )

* **Gaussian**
	* Descrição: 
	* Argumentos: (prob=unive) )

* **Edge**
	* Descrição: 
	* Argumentos: (prob=unive) )

* **Grey_Erosion**
	* Descrição: 
	* Argumentos: (prob=unive) )

## Usando a função ```utils.call_thread()```
A forma mais comoda de se utilizar é passando o objeto pipe criado para a função
```python 
utils.call_thread()
```
Essa função tambem pode ser executada por mais de uma thread divindo a base entre elas, onde cada uma vai realizar as operações e ao final serão reunidas e serão o retorno da função, reunidas com as imagens originais

Exemplo:
```python
resultado = utils.call_thread(img_per_thread = 100, data = data, pipe_instance = pipe_example, image_per_image = 2)
```
#### Os argumentos de ```utils.call_thread()``` são:
* **img_per_thread**: Numero de imagens que serão passadas para cada thread (caso a divisão não seja exata uma thread recebera um numero menor de imagens)
* **data**: Base que sera usada, deve esta na forma de um vetor (lista) [imagem, classe]
* **pipe_instance**: pipe que sera utilizado
* **image_per_image**: quantas imagens geradas para cada imagem original da base
#### Saida:
É um vetor (lista) na forma [imagem, classe], contendo as novas imagens geradas e as imagens originais (como são atribuidas probabilidades para as operações é possivel que nenhuma seja aplicada, neste caso a imagem não sera adicionada podendo gerar um numero de imagens menor que o esperado).

#### A ser adicionado
* salvar imagens geradas
* opção de retornar apenas imagens geradas
* opção adicionar ou não imagens inalteradas
