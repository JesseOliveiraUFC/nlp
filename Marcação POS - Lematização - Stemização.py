import nltk 
import spacy 

pln = spacy.load('pt_core_news_sm')

doc = pln('Sou Jessé Oliveira e estou aprendendo Processamento de Linguagem Natural')

print(type(doc))

# Verifica as classes gramaticais de cada palavra do texto.
for token in doc:
    print(token.text, token.pos_)
    
    
# LEMATIZAÇÃO E STEMIZAÇÃO
    # LEMATIZAÇÂO
    # Serve para reduzir a base de dados eliminando palavras que 
    # são parecidas (possuem o mesmo radical)
for token in doc:
    print(token.text, token.lemma_)

#Exemplo 2    -- Muito utilizado para ChatBot para entendimentodo que o usuário está falando
doc2 = pln('encontrei encontraram encontrarão encontrariam cursando curso cursei')
[token.lemma_ for token in doc2]


    # STEMIZAÇÂO
nltk.download('rslp')   # Faz a instalação do Stemmer  

stemmer = nltk.stem.RSLPStemmer()
stemmer.stem('aprender')

for token in doc:
    print(token.text, token.lemma_, stemmer.stem(token.text))