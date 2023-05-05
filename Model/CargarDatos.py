import pandas as pd
     
def cargarDatos(ruta):
     pd.set_option('display.max_colwidth', None)
     # Leemos los datos
     df = pd.read_csv(ruta, index_col=None)
     return df

