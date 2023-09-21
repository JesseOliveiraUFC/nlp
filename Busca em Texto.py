import spacy
from spacy.matcher import PhraseMatcher
import web_scrapping_nlp as web

# Faz a leitura do text usando o arquivo web_scrapping_nlp.py
conteudo = web.main()

# Carrega a biblioteca
pln = spacy.load('pt_core_news_sm')

# Fazendo uma busca da palavra no texto
string = 'turing'
token_pesquisa = pln(string)

# Criando o matcher 
matcher = PhraseMatcher(pln.vocab)
matcher.add('SEARCH', None, token_pesquisa)

# Iniciando o identificador
doc = pln(conteudo)
matches = matcher(doc)
print(matches)

# Melhorando a visualização da busca
print(f"Resultados encontrados: {len(matches)}")

num_palavras = 50
texto = ''
print(string.upper(),"\n")
print(f"Quantidade de palavras encontradas: {len(matches)} \n\n")
for i in matches:
    inicio = i[1] - num_palavras
    if inicio < 0:
        inicio = 0
    texto += str(doc[inicio:i[2] + num_palavras]).replace(string, f"{string.upper()}")
    texto += '\n\n'
print(texto)
