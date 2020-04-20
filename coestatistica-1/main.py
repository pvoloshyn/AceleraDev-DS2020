import pandas as pd
import numpy as np

def mode(s):
    """
    Retorna a moda de um Pandas Series

    :params s: objeto do tipo pd.Series
    :returns: moda
    """
    return s.value_counts().index[0]

def main():
    # lê dataframe e coloca RowNumber como índice
    df = pd.read_csv('desafio1.csv', index_col='RowNumber')

    # faz os devidos agrupamentos pelo estado
    grouped = df.groupby('estado_residencia').agg(
        {'pontuacao_credito': [
            mode,
            np.mean,
            np.median,
            np.std]
        }).reset_index()

    # renomeamos as colunas
    grouped.columns = ['estado_residencia', 'moda', 'media', 'mediana', 'desvio_padrao']
    # como vamos transpor a tabela para facilitar a exportação, configuramos o índice como sendo o estado
    grouped.set_index('estado_residencia', inplace=True)
    # exportamos o resultado conforme esperado
    grouped.transpose().to_json('submission.json')

if __name__ == '__main__':
    main()