import spacy
import web_scrapping_nlp as web

# Importando o mapa de cores e definindo quais serão utilizadas
from matplotlib.colors import ListedColormap
color_map = ListedColormap(['orange', 'green', 'red', 'magenta'])

# Criando a Nuvem de Palavras
from wordcloud import WordCloud 
cloud = WordCloud(background_color = 'white', max_words = 100, colormap = color_map)

# Biblioteca para visualização dos gráficos
import matplotlib.pyplot as plt 

# Biblioteca para remover as palavras genéricas (de, da, do, para, que, e, etc...)
from spacy.lang.pt.stop_words import STOP_WORDS

# Carrega a biblioteca
pln = spacy.load('pt_core_news_sm')

# Faz a leitura do texto usando o arquivo web_scrapping_nlp.py
conteudo = web.main()

# Removendo as Stop Words
doc = pln(conteudo)   # Convertendo a string para formato de lista
lista_token = []
for token in doc:
    lista_token.append(token.text)

sem_stop_words = []
for palavra in lista_token:
    if pln.vocab[palavra].is_stop == False:
        sem_stop_words.append(palavra)

# Realiza a criação da nuvem de palavras
cloud = cloud.generate(' '.join(sem_stop_words))
plt.figure(figsize=(15,15))
plt.imshow(cloud)
plt.axis('off')
plt.show()

