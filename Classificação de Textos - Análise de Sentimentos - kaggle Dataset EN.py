# Usando somente o Spacy
import spacy 
import pandas as pd
import string
import random 
import seaborn as sns 
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.training import Example
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# Constantes
CATEGORIAS = ["JOY", "SADNESS", "ANGER", "FEAR", "LOVE", "SURPRISE"]
EPOCAS = 1000
TAMANHO_BATCH = 30

# Criando o modelo
pln = spacy.load('en_core_web_lg')

# Realizando a leitura dos dataframes
train = pd.read_csv("Classificação de Sentimentos - Base 2 EN/train.txt", sep = ';', header=None, names = ["phrase","emotion"])
test = pd.read_csv("Classificação de Sentimentos - Base 2 EN/test.txt", sep = ';', header=None, names = ["phrase","emotion"])

train.head(10)

# Imprime a quantidade de dados de cada categoria
test['emotion'].value_counts()
# Plota um gráfico com as quantidades de cada categoria
sns.countplot(x='emotion', data=train)

#### -------------- Pré-processamento do texto -------------------

# Remoção de pontuações, stop words e dígitos numéricos
pontuacoes = string.punctuation
#print(pontuacoes)

# Função que realiza o pré processamento do texto
def preprocess(texto):
    doc = pln(texto.lower())
    lista = []
    
    for token in doc:  # tokenização
        #lista.append(token.text)
        lista.append(token.lemma_)  # Usando a lematização
        
    lista = [palavra for palavra in lista if palavra not in STOP_WORDS and palavra not in pontuacoes]
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])
    return lista

# Aplicando o pré-processamento nas bases de treinamento e teste
train['phrase'] = train['phrase'].apply(preprocess)
test['phrase'] = test['phrase'].apply(preprocess)

## Criando a função de normalização da base de dados para ficar no padrão necesário para o modelo
def structreatment(df):
    mapping = {
        'joy': {'JOY': True, 'SADNESS': False, 'ANGER': False, 'FEAR': False, 'LOVE': False, 'SURPRISE': False},
        'sadness': {'JOY': False, 'SADNESS': True, 'ANGER': False, 'FEAR': False, 'LOVE': False, 'SURPRISE': False},
        'anger': {'JOY': False, 'SADNESS':False, 'ANGER':True, 'FEAR':False, 'LOVE':False, 'SURPRISE':False},
        'fear': {'JOY': False, 'SADNESS':False, 'ANGER':False, 'FEAR':True, 'LOVE':False, 'SURPRISE':False},
        'love': {'JOY': False, 'SADNESS':False, 'ANGER':False, 'FEAR':False, 'LOVE':True, 'SURPRISE':False},
        'surprise': {'JOY': False, 'SADNESS':False, 'ANGER':False, 'FEAR':False, 'LOVE':False, 'SURPRISE':True}
    }
    return [[phrase, mapping[emotion].copy()] for phrase, emotion in zip(df['phrase'], df['emotion'])]    
   
# Aplicando a normalização nas bases: 
base_train = structreatment(train)
base_test = structreatment(test)


#_______________________________________________________________________________________________________________    
####### Definindo o Classificador SPACY
# Crie um modelo em branco para a língua inglesa com as categorias definidas previamente
def create_model(categorias):
    modelo = spacy.blank('en')
    textcat = modelo.add_pipe('textcat')
    for cat in categorias:
        textcat.add_label(cat)
    return modelo

modelo = create_model(CATEGORIAS)

historico = []
    
##### Iniciando o treinamento da rede
# Refatoração do treinamento
def train_model(modelo, base_train, epocas=EPOCAS, tamanho_batch=TAMANHO_BATCH):
    historico = []
    modelo.begin_training()
    for epoca in range(epocas):
        random.shuffle(base_train)
        losses = {}
        for batch in spacy.util.minibatch(base_train, tamanho_batch):
            examples = [Example.from_dict(modelo.make_doc(texto), {"cats": entities}) for texto, entities in batch]
            modelo.update(examples, losses=losses)
        if epoca % 100 == 0:
            print(losses)
            historico.append(losses)
    return historico

historico = train_model(modelo, base_train)
    
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
modelo.to_disk("modelo_2-en")
    
###### FAZENDO A LEITURA DA REDE NEURAL TREINADA
#modelo_treinado = spacy.load("modelo_2-en")

##### AVALIAÇÃO DO MODELO 

### -------- Validação na base de testes
# Valide as previsões em lote
def batch_predict(modelo, texts):
    docs = list(modelo.pipe(texts))
    return [doc.cats for doc in docs]

previsoes = batch_predict(modelo, test['phrase'])
    
# Realizando tratamento para fazer a comparação, selecionando a chave que apresenta maior valor (probabilidade):
previsoes_final = []
for previsao in previsoes:
    previsoes_final.append(max(previsao, key = previsao.get).lower())  # Faz a varredura no dicionário e coleta a chaave com maior valor
        
previsoes_final = np.array(previsoes_final)

# Realizando a comparação com a base de dados

# Calculando a confusion_matrix e a accuracy
accuracy = accuracy_score(test['emotion'], previsoes_final)
unique_labels = test['emotion'].unique()
cm = confusion_matrix(test['emotion'], previsoes_final, labels=unique_labels)

# Imprimindo a accuracy
print(accuracy)

# Visualizando a confusion_matrix usando Seaborn
plt.figure(figsize=(10, 7))  # Define o tamanho da figura
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=CATEGORIAS, yticklabels=CATEGORIAS)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

