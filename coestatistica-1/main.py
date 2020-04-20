import pandas as pd
import numpy as np

def main():
    # lê dataframe e coloca RowNumber como índice
    df = pd.read_csv('desafio1.csv', index_col='RowNumber')

    # faz os devidos agrupamentos pelo estado
    grouped = df.groupby('estado_residencia').agg(
        {'pontuacao_credito': [
            pd.Series.mode,
            np.mean,
            np.median,
            np.std]
        })

    # renomeamos as colunas
    grouped.columns = ['moda', 'media', 'mediana', 'desvio_padrao']
    # exportamos o resultado conforme esperado
    grouped.to_json('submission.json', orient='index')

if __name__ == '__main__':
    main()