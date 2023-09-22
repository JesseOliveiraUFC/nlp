# Usando somente o Spacy
import spacy 
import pandas as pd
import string
import random 
import seaborn as sns 
import numpy as np
from spacy.lang.pt.stop_words import STOP_WORDS
from spacy.training import Example
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# Criando o modelo
pln = spacy.load('pt_core_news_sm')

# Realizando a leitura dos dataframes
trainning = pd.read_csv("Classificação de Sentimentos/base_treinamento.csv")
test = pd.read_csv("Classificação de Sentimentos/base_teste.csv")

test.head(10)

# Imprime a quantidade de dados de cada categoria
trainning['sentimento'].value_counts()
# Plota um gráfico com as quantidades de cada categoria
sns.countplot(x='sentimento', data=trainning)

#### -------------- Pré-processamento do texto -------------------

# Remoção de pontuações, stop words e dígitos numéricos
pontuacoes = string.punctuation
#print(pontuacoes)

stop_words = STOP_WORDS
#print(stop_words)

# Função que realiza o pré processamento do texto
def preprocess(texto):
    doc = pln(texto.lower())
    lista = []
    
    for token in doc:  # tokenização
        #lista.append(token.text)
        lista.append(token.lemma_)  # Usando a lematização
        
    lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in pontuacoes]
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])
    return lista

# Aplicando o pré-processamento nas bases de treinamento e teste
trainning['frase'] = trainning['frase'].apply(preprocess)
test['frase'] = test['frase'].apply(preprocess)

## Realizando a normalização da base de dados para ficar no padrão necesário para o modelo
base_train = []
for frase, sentimento in zip(trainning['frase'], trainning['sentimento']):
    if sentimento == 'alegria':
        dic = ({'ALEGRIA': True, 'MEDO':False})
    elif sentimento == 'medo':
        dic = ({'ALEGRIA': False, 'MEDO':True})
        
    base_train.append([frase, dic.copy()])

base_test = []
for frase, sentimento in zip(test['frase'], test['sentimento']):
    if sentimento == 'alegria':
        dic = ({'ALEGRIA': True, 'MEDO':False})
    elif sentimento == 'medo':
        dic = ({'ALEGRIA': False, 'MEDO':True})
        
    base_test.append([frase, dic.copy()])
    
    
####### Definindo o Classificador SPACY
# Crie um modelo em branco para a língua portuguesa
modelo = spacy.blank('pt')

# Adicione o componente 'textcat' ao pipeline do modelo
modelo.add_pipe('textcat')

# Obtenha o componente 'textcat' para adicionar labels
categorias = modelo.get_pipe('textcat')
categorias.add_label("ALEGRIA")
categorias.add_label("MEDO")

historico = []
    
##### Iniciando o treinamento da rede
modelo.begin_training()

for epoca in range(1000):
    random.shuffle(base_train)
    losses = {}
    for batch in spacy.util.minibatch(base_train, 30):
        examples = [Example.from_dict(modelo.make_doc(texto), {"cats": entities}) for texto, entities in batch]
        modelo.update(examples, losses=losses)
    if epoca % 100 == 0:
        print(losses)
        historico.append(losses)
    
##### Verificando a variação temporal do erro:
    
#print(historico)
historico_loss = []
for i in historico:
    historico_loss.append(i.get('textcat'))
historico_loss = np.array(historico_loss)

plt.plot(historico_loss)
plt.title('Progressão do Erro')
plt.xlabel('Épocas (x100)')
plt.ylabel('Erro')

####### SALVAR O MODELO TREINADO
modelo.to_disk("modelo")
    
###### FAZENDO A LEITURA DA REDE NEURAL TREINADA
modelo_treinado = spacy.load("modelo")

##### Fazendo a previsão com duas frases distintas e fazendo o pré-processamento
texto_alegria = "Hoje é um dos melhores dias da minha vida; sinto-me nas nuvens!"   
texto_alegria = preprocess(texto_alegria)
texto_medo = "O som misterioso vindo do porão à meia-noite me deixou aterrorizado." 
texto_medo = preprocess(texto_medo)

previsao1 = modelo_treinado(texto_alegria)
print(previsao1.cats)

previsao2 = modelo_treinado(texto_medo)
print(previsao2.cats)

##### AVALIAÇÃO DO MODELO 

### -------- Avaliação na base de treinamento

previsoes = []
for texto in trainning['frase']:
    previsao = modelo_treinado(texto)
    previsoes.append(previsao.cats)
    
# Realizando tratamento para fazer a comparação:
previsoes_final = []
for previsao in previsoes:
    if previsao['ALEGRIA'] > previsao['MEDO']:
        previsoes_final.append("alegria")
    else:
        previsoes_final.append("medo")
        
previsoes_final = np.array(previsoes_final)

# Realizando a comparação com a base de dados

print(accuracy_score(trainning['sentimento'], previsoes_final))
cm = confusion_matrix(trainning['sentimento'], previsoes_final)
print(cm)

### -------- Avaliação na base de testes
previsoes = []
for texto in test['frase']:
    previsao = modelo_treinado(texto)
    previsoes.append(previsao.cats)
    
# Realizando tratamento para fazer a comparação:
previsoes_final = []
for previsao in previsoes:
    if previsao['ALEGRIA'] > previsao['MEDO']:
        previsoes_final.append("alegria")
    else:
        previsoes_final.append("medo")
        
previsoes_final = np.array(previsoes_final)

# Realizando a comparação com a base de dados

print(accuracy_score(test['sentimento'], previsoes_final))
cm = confusion_matrix(test['sentimento'], previsoes_final)
print(cm)

