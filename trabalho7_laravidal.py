# Lara Franco Chaves Vidal
# Trabalho 7 - Construção de Interpretadores
# Professor Frank Alcântara

# Sua tarefa será gerar uma matriz de distância, computando o cosseno do ângulo entre todos os 
# vetores que encontramos usando o tf-idf. 

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import math
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

print("Imports feitos")

# SITE 1
page1 = "https://www.sas.com/en_us/insights/analytics/what-is-natural-language-processing-nlp.html"
html = requests.get(page1).text
soup1 = BeautifulSoup(html, "html.parser")
 
text_array1 = []
get = soup1.find_all('p')
text_dir1 = list(get)

for text in text_dir1:
    text_array1.append(text.get_text().split(" "))

# Removendo as palavras repetidas dentro do site 1

all1 = []
for i in text_array1:
  for j in i:
    all1.append(j)

vocab1 = []

contador = 0

for word in all1:
    if word not in vocab1:
        vocab1.append(word)
        contador += all1.count(word)
    contador = 0
    
# SITE 2
page2 = "https://www.datarobot.com/blog/what-is-natural-language-processing-introduction-to-nlp/"
html = requests.get(page2).text
soup2 = BeautifulSoup(html, "html.parser")
 
text_array2 = []
get = soup2.find_all('p')
text_dir2 = list(get)

for text in text_dir2:
    text_array2.append(text.get_text().split(" "))

# Removendo as palavras repetidas dentro do site 2

all2 = []
for i in text_array2:
  for j in i:
    all2.append(j)

vocab2 = []
contador = 0

for word in all2:
    if word not in vocab2:
        vocab2.append(word)
        contador += all2.count(word)
    contador = 0
    
# SITE 3
page3 = "https://hbr.org/2022/04/the-power-of-natural-language-processing"
html = requests.get(page3).text
soup3 = BeautifulSoup(html, "html.parser")
 
text_array3 = []
get = soup3.find_all('p')
text_dir3 = list(get)

for text in text_dir3:
    text_array3.append(text.get_text().split(" "))

# Removendo as palavras repetidas dentro do site 3

all3 = []
for i in text_array3:
  for j in i:
    all3.append(j)

vocab3 = []
contador = 0

for word in all3:
    if word not in vocab3:
        vocab3.append(word)
        contador += all3.count(word)
    contador = 0
    
# SITE 4
page4 = "https://monkeylearn.com/natural-language-processing/"
html = requests.get(page4).text
soup4 = BeautifulSoup(html, "html.parser")
 
text_array4 = []
get = soup4.find_all('p')
text_dir4 = list(get)

for text in text_dir4:
    text_array4.append(text.get_text().split(" "))

# Removendo as palavras repetidas dentro do site 4

all4 = []
for i in text_array4:
  for j in i:
    all4.append(j)

vocab4 = []
contador = 0

for word in all4:
    if word not in vocab4:
        vocab4.append(word)
        contador += all4.count(word)
    contador = 0
    
# SITE 5
page5 = "https://www.tableau.com/learn/articles/natural-language-processing-examples"
html = requests.get(page5).text
soup5 = BeautifulSoup(html, "html.parser")
 
text_array5 = []
get = soup5.find_all('p')
text_dir5 = list(get)

for text in text_dir5:
    text_array5.append(text.get_text().split(" "))

# Removendo as palavras repetidas dentro do site 5

all5 = []
for i in text_array5:
  for j in i:
    all5.append(j)

vocab5 = []
contador = 0

for word in all5:
    if word not in vocab5:
        vocab5.append(word)
        contador += all5.count(word)
    contador = 0
    
# Adicionando os lexemas de todos os sites, sem repetições 
all_texts = []
vocabSites = [vocab1, vocab2, vocab3, vocab4, vocab5]

for site in vocabSites:
  for word in site:
      if word not in all_texts:
          all_texts.append(word)

# Calculando a frequência que cada palavra de todos os sites aparece em cada site individualmente
wordFreq1 = []
wordFreq2 = []
wordFreq3 = []
wordFreq4 = []
wordFreq5 = []
contador = 0

for word in all_texts:
    contador += all1.count(word)
    wordFreq1.append(contador)
    contador = 0

for word in all_texts:
    contador += all2.count(word)
    wordFreq2.append(contador)
    contador = 0

for word in all_texts:
    contador += all3.count(word)
    wordFreq3.append(contador)
    contador = 0

for word in all_texts:
    contador += all4.count(word)
    wordFreq4.append(contador)
    contador = 0

for word in all_texts:
    contador += all5.count(word)
    wordFreq5.append(contador)
    contador = 0
    
# Criando a tabela com a biblioteca pandas
from google.colab.data_table import DataTable
DataTable.max_columns = 2960

# Contabilizando quantos termos tem em cada documento
termos1 = 0
termos2 = 0
termos3 = 0
termos4 = 0
termos5 = 0

for termo in all1:
    termos1 += 1
for termo in all2:
    termos2 += 1
for termo in all3:
    termos3 += 1
for termo in all4:
    termos4 += 1
for termo in all5:
    termos5 += 1
    
# Fazendo o cálculo do TF
tf1 = []
for i in wordFreq1:
      x = wordFreq1[i] / termos1
      tf1.append(round(x, 4))

tf2 = []
for i in wordFreq2:
      x = wordFreq2[i] / termos2
      tf2.append(round(x, 4))

tf3 = []
for i in wordFreq3:
      x = wordFreq3[i] / termos3
      tf3.append(round(x, 4))

tf4 = []
for i in wordFreq4:
      x = wordFreq4[i] / termos4
      tf4.append(round(x, 4))

tf5 = []
for i in wordFreq5:
      x = wordFreq5[i] / termos5
      tf5.append(round(x, 4))
      
# Fazendo o cálculo do IDF

idf = []
numDocumentos = 5
y = 0

for termo in all_texts:
    for site in vocabSites:
        if termo in site:
            y += 1
    valor = round(math.log10(numDocumentos/y), 4)
    idf.append(valor)
    y = 0    
    
# Fazendo o cálculo do TF-IDF

contador1 = 0
contador2 = 0
contador3 = 0
contador4 = 0
contador5 = 0

tf_idf1 = []
for termo in all_texts:
      valor = round((tf1[contador1] * idf[contador1]), 4)
      tf_idf1.append(valor)
      contador1 += 1

tf_idf2 = []
for termo in all_texts:
      valor = round((tf2[contador2] * idf[contador2]), 4)
      tf_idf2.append(valor)
      contador2 += 1

tf_idf3 = []
for termo in all_texts:
      valor = round((tf3[contador3] * idf[contador3]), 4)
      tf_idf3.append(valor)
      contador3 += 1

tf_idf4 = []
for termo in all_texts:
      valor = round((tf4[contador4] * idf[contador4]), 4)
      tf_idf4.append(valor)
      contador4 += 1
      
tf_idf5 = []
for termo in all_texts:
      valor = round((tf5[contador5] * idf[contador5]), 4)
      tf_idf5.append(valor)
      contador5 += 1
      
# Tabela

coluna1 = pd.Series({all_texts[i]: tf_idf1[i] for i in range(len(all_texts))})
coluna2 = pd.Series({all_texts[i]: tf_idf2[i] for i in range(len(all_texts))})
coluna3 = pd.Series({all_texts[i]: tf_idf3[i] for i in range(len(all_texts))})
coluna4 = pd.Series({all_texts[i]: tf_idf4[i] for i in range(len(all_texts))})
coluna5 = pd.Series({all_texts[i]: tf_idf5[i] for i in range(len(all_texts))})

df = pd.DataFrame([coluna1, coluna2, coluna3, coluna4, coluna5])
df.index = np.arange(1, len(df) + 1)
df.index.names = ['DOCUMENTOS']
df

# Matriz de distância

matrizDistancia = []
vetor1 = []
vetor2 = []
vetor3 = []
vetor4 = []
vetor5 = []
all_sentences = [tf_idf1, tf_idf2, tf_idf3, tf_idf4, tf_idf5]

def fazMatriz(vetor):
    lista = []
    
    i = 0 #indice em todas_sentencas
    a = vetor
    while i < 5:
        b = all_sentences[i]
        i += 1
        cos_sim = np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
        lista.append(cos_sim)
    return lista

vetor1 = fazMatriz(tf_idf1)
vetor2 = fazMatriz(tf_idf2)
vetor3 = fazMatriz(tf_idf3)
vetor4 = fazMatriz(tf_idf4)
vetor5 = fazMatriz(tf_idf5)

# table matriz distancia

coluna1 = pd.Series(vetor1)
coluna2 = pd.Series(vetor2)
coluna3 = pd.Series(vetor3)
coluna4 = pd.Series(vetor4)
coluna5 = pd.Series(vetor5)

md = pd.DataFrame([coluna1, coluna2, coluna3, coluna4, coluna5])
md.index = np.arange(1, len(df) + 1)
md.index.names = ['VETORES']
md
