# Image-Augmentation
--EM DESENVOLVIMENTO--

Uma biblioteca para aumentar bases de dados de imagens, nela se cria um modelo...
## Utilização
Para utilizar a biblioteca
Primeiro se importa biblioteca
```python
import Aug
```
Declara-se o modelo
```python
pipe = Pipe()
```
Usa-se o metodo .add para adicionar as operações que serão utilizadas (em ordem)
```python
pipe.add(Random_Noise(0.7))
pipe.add(Random_Erasing(prob=1, rectangle_area=0.3, repetitions=2))
```
Passa a imagem e o numero de imagens a serem geradas como argumento da função operar, que retorna uma lista com as imagens geradas
```python
lista_de_imagens = []
lista_de_imagens = pipe.operar(image, 3)
```





## Operações
##### Toda operação vai receber um argumento " prob " que indica a probabilidade da operação ser aplicada ou não 
* Rotação
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
pipe.add(Rotacao(0.2, max_left_rotation=40, max_right_rotation=40))
```
			
* Shift
	* Descrição
	* Argumentos
		* 
* Tilt
	* Descrição
	* Argumentos
		* 
