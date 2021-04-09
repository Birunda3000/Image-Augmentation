# Image-Augmentation
Uma biblioteca para aumentar bases de dados de imagens
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
* Rotação
	* Descrição
	* Argumentos
		* 
* Shift
	* Descrição
	* Argumentos
		* 
* Tilt
	* Descrição
	* Argumentos
		* 
