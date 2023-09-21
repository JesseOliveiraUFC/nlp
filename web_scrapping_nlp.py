import bs4 as bs
import urllib.request

def main():
    # FAZENDO WEB SCRAPING PARA PERGAR OS DADOS DE UM ARTIGO DA WIKIPEDIA
    url = 'https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial'
    
    # Usa a biblioteca urllib para acessar a pagina
    dados = urllib.request.urlopen(url)
    # Faz a leitura dos dados da página # Extrai o código HTML
    dados = dados.read()
    #print(dados)
    
    # Torna a leitura do HTML mais "bonitinha"
    dados_html = bs.BeautifulSoup(dados, 'lxml')
    #print(dados_html)
    
    # Faz a coleta de todas as tags definidas - Neste caso, são as tags <p>
    paragrafos = dados_html.find_all('p')
    # Mostra somente os textos, sem as tags
    #print(paragrafos[0].text)
    
    # Concatena todo o texto da página em uma única string
    conteudo = ''
    for p in paragrafos:
        conteudo += p.text
    
    # É recomendado sempre trabalhar com letras minúsculas em NLP
    conteudo = conteudo.lower()    
    
    #print(conteudo)
    return conteudo

if __name__ == '__main__':
    main()

