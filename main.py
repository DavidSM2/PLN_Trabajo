from Model.CargarDatos import cargarDatos
from Model.PreprocesadoTexto import normalizarDoc
def main():
    df = cargarDatos('./Datos_test/sem_eval_train_es.csv')
    print(df['Tweet'])
    print(normalizarDoc(df['Tweet']))







if __name__ == "__main__":
    main()