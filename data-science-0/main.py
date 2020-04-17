#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[4]:


def q1():
    return black_friday.shape

q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[42]:


def q2():
    filt_gender = black_friday['Gender']=='F'
    filt_age = black_friday['Age']=='26-35'
    result = (filt_gender & filt_age).sum()
    return int(result)

q2()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[43]:


def q3():
    result = black_friday['User_ID'].nunique()
    return int(result)

q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[44]:


def q4():
    result = black_friday.dtypes.nunique()
    return int(result)

q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[45]:


def q5():
    result = black_friday.isna().any(axis=1).mean()
    return float(result)

q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[46]:


def q6():
    result = black_friday.isna().sum(axis=0).max()
    return int(result)

q6()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[28]:


def q7():
    return black_friday['Product_Category_3'].dropna().value_counts().index[0]

q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[62]:


def q8():
    purchases = black_friday['Purchase']
    purchases_norm = (purchases - purchases.min()) / (purchases.max() - purchases.min())
    result = purchases_norm.mean()
    return round(result, 3)

q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[51]:


def q9():
    purchases = black_friday['Purchase']
    purchases_norm = (purchases - purchases.mean()) / purchases.std()
    result = purchases_norm.between(-1, 1).sum()
    return int(result)

q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[49]:


def q10():
    pc2_na = black_friday['Product_Category_2'].isna()
    vc = black_friday.loc[pc2_na, 'Product_Category_3'].value_counts()
    return len(vc)==0

q10()


# In[ ]:




