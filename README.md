# libras-alphabet-recognition
Repositório destinado para a crição de um modelo de Redes Neurais Convolucionais para reconhecimento do alfabeto em LIBRAS. Projeto totalmente prototipado.

## ARQUITETURA USADA: 

INPUT => CONV => POOL => CONV => POOL => CONV => POOL => FC => FC => OUTPUT 

## ARQUITETURA DOS ARQUIVOS

- datasets/: Contém a base de dados utilizada para o treinamento da CNN
- main/: Carrega os códigos principais do projeto, incluindo o da arquitetura da CNN (model.py) e do treinamento e parametrização do modelo (training.py)
- models/: Contém os modelos e os insigts gerados com o treinamento
- results/: Carrega os resultados do treinamento dos modelos

## REFERÊNCIAS

1. http://cs231n.github.io/convolutional-networks/
