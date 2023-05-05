import re
import spacy


nlp = spacy.load("es_core_news_lg")

acentos = [
    ('á', 'a'), ('à', 'a'), ('ä', 'a'), ('â', 'a'),
    ('é', 'e'), ('è', 'e'), ('ë', 'e'), ('ê', 'e'),
    ('í', 'i'), ('ì', 'i'), ('ï', 'i'), ('î', 'i'),
    ('ó', 'o'), ('ò', 'o'), ('ö', 'o'), ('ô', 'o'),
    ('ú', 'u'), ('ù', 'u'), ('ü', 'u'), ('û', 'u')
]


def normalizarDoc(doc):
    doc_norm = []
    for tweet in doc:
        tweet_norm = _normalizarTexto(tweet)
        doc_norm.append(tweet_norm)
    return doc_norm


def _normalizarTexto(texto):
    # separamos después de ciertos signos de puntuación
    texto = re.sub(r"([\.\?])", r"\1 ", texto)
    #Quitamos numeros, saltos de linea
    texto = re.sub(r'(\d+|\n)','',texto)
    #Quitamos acentos o dieresis
    for letraAccent, letraNorm in acentos:
        texto = re.sub(r'{0}'.format(letraAccent), letraNorm, texto)
    #Separamos los puntos, comas y menciones
    texto = _normalizarPuncts(texto)
    texto = _normalizarCaracteristicasTwets(texto)
    '''doc = nlp(texto)
    tokens = [t.lemma_.lower() if not t.ent_type_ == 'PER' else '_PERSONA_'
              for t in doc if not t.is_punct and not t.is_stop and not t.is_space and len(t.text) > 1]
    salida = ' '.join(tokens)'''
    return texto

def _normalizarPuncts(texto):
    texto = re.sub(r'(?P<punct>[\.\,])(?P<simbol>[\@\#\¿\?\¡\!\(\)])', r'\g<punct> \g<simbol>', texto)
    return texto

def _normalizarCaracteristicasTwets(texto):
    texto = re.sub(r' @\S+', ' MENCION_TWEET', texto)
    texto = re.sub(r' #\S+', ' HASTAG_TWEET', texto)
    return texto
