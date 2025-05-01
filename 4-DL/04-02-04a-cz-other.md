# Word Embeddings a jazykové modely

## Obsah
1. Úvod do word embeddings
2. Tradiční metody reprezentace slov
3. Word2Vec
4. GloVe embeddings
5. FastText
6. Kontextové embeddings
7. Základy jazykových modelů
8. N-gram jazykové modely
9. Neuronové jazykové modely
10. Implementace vlastního jazykového modelu
11. Praktické aplikace
12. Shrnutí a nejlepší postupy

## Úvod do word embeddings

Word embeddings jsou technika pro převod slov do vektorového prostoru, kde podobná slova jsou reprezentována podobnými vektory. Tato reprezentace umožňuje zachytit sémantické a syntaktické vztahy mezi slovy, což je klíčové pro zpracování přirozeného jazyka (NLP).

**Klíčové vlastnosti word embeddings:**
- Převod slov na hustě vektorové reprezentace s plovoucí desetinnou čárkou
- Zachycení sémantické podobnosti pomocí geometrické blízkosti ve vektorovém prostoru
- Zachycení lingvistických a kontextových vlastností slov
- Možnost provádění algebraických operací na slovech (např. "král" - "muž" + "žena" ≈ "královna")

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Ukázka vizualizace word embeddings

# Zjednodušený příklad word embedding vektorů
word_vectors = {
    'král': np.array([0.3, 0.8, 0.1, 0.5]),
    'královna': np.array([0.4, 0.7, 0.8, 0.3]),
    'muž': np.array([0.1, 0.2, 0.3, 0.4]),
    'žena': np.array([0.2, 0.1, 0.9, 0.2]),
    'pes': np.array([0.9, 0.2, 0.3, 0.8]),
    'kočka': np.array([0.8, 0.3, 0.2, 0.7])
}

# Vytvoření matice vektorů a seznamu odpovídajících slov
words = list(word_vectors.keys())
vectors = np.array([word_vectors[word] for word in words])

# Redukce dimenzí pro vizualizaci pomocí PCA
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Vizualizace vektorů
plt.figure(figsize=(10, 6))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='b', alpha=0.7)

# Přidání popisků slov
for i, word in enumerate(words):
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=12)

plt.title('Vizualizace word embeddings pomocí PCA')
plt.xlabel('Komponenta 1')
plt.ylabel('Komponenta 2')
plt.grid(True)
plt.show()

# Demonstrace vektorových operací
# Král - Muž + Žena ≈ Královna
analogy_vector = word_vectors['král'] - word_vectors['muž'] + word_vectors['žena']
distances = {w: np.linalg.norm(word_vectors[w] - analogy_vector) for w in word_vectors}
sorted_distances = sorted(distances.items(), key=lambda x: x[1])

print("Výsledek analogie 'král - muž + žena':")
for word, dist in sorted_distances:
    print(f"{word}: vzdálenost = {dist:.4f}")
```

## Tradiční metody reprezentace slov

Před nástupem word embeddings byly texty reprezentovány pomocí jednodušších metod:

### One-Hot Encoding

V one-hot kódování je každé slovo reprezentováno vektorem délky slovníku, kde pouze jeden prvek (odpovídající pozici daného slova ve slovníku) má hodnotu 1, zbytek 0.

```python
import numpy as np

# Jednoduchý slovník
vocabulary = {'pes': 0, 'kočka': 1, 'myš': 2, 'sýr': 3}
vocab_size = len(vocabulary)

# One-hot kódování
def one_hot_encode(word, vocabulary):
    encoding = np.zeros(len(vocabulary))
    encoding[vocabulary[word]] = 1
    return encoding

# Ukázka one-hot kódování
for word in vocabulary:
    one_hot = one_hot_encode(word, vocabulary)
    print(f"{word}: {one_hot}")
```

### Bag of Words (BoW)

Metoda Bag of Words reprezentuje dokument jako vektor četnosti slov bez ohledu na jejich pořadí.

```python
from sklearn.feature_extraction.text import CountVectorizer

# Ukázkové dokumenty
documents = [
    "Pes honí kočku",
    "Kočka loví myš",
    "Myš jí sýr",
    "Pes štěká na kočku"
]

# Bag of Words model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# Získání slovníku a matice četnosti
vocabulary = vectorizer.get_feature_names_out()
bow_matrix = X.toarray()

print("Slovník:")
print(vocabulary)
print("\nBag of Words reprezentace:")
for i, doc in enumerate(documents):
    print(f"Dokument {i+1}: {doc}")
    print(f"Vektor: {bow_matrix[i]}")
```

### TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF je metoda, která přiřazuje váhy slovům na základě jejich četnosti v dokumentu a vzácnosti napříč kolekcí dokumentů.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Použití stejných dokumentů jako v předchozím příkladu
documents = [
    "Pes honí kočku",
    "Kočka loví myš",
    "Myš jí sýr",
    "Pes štěká na kočku"
]

# TF-IDF model
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(documents)

# Získání slovníku a matice TF-IDF
vocabulary = tfidf_vectorizer.get_feature_names_out()
tfidf_matrix = X_tfidf.toarray()

print("Slovník:")
print(vocabulary)
print("\nTF-IDF reprezentace:")
for i, doc in enumerate(documents):
    print(f"Dokument {i+1}: {doc}")
    print(f"Vektor: {tfidf_matrix[i]}")
```

## Word2Vec

Word2Vec je jedním z nejpopulárnějších algoritmů pro vytváření word embeddings. Existují dvě hlavní architektury:
- **CBOW (Continuous Bag of Words)** - Předpovídá cílové slovo na základě kontextových slov
- **Skip-gram** - Předpovídá kontextová slova na základě cílového slova

### Implementace Word2Vec pomocí Gensim

```python
import gensim
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Stažení potřebných dat pro tokenizaci
nltk.download('punkt')

# Příprava dat
sentences = [
    "Král vládne v královském paláci",
    "Královna nosí krásnou korunu",
    "Praha je hlavní město České republiky",
    "Prezident žije na Pražském hradě",
    "Programování v Pythonu je zábavné",
    "Umělá inteligence se rychle vyvíjí",
    "Neuronové sítě se používají pro strojové učení",
    "Pes je nejlepší přítel člověka",
    "Kočky mají devět životů"
]

# Tokenizace vět
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Trénování Word2Vec modelu
# sg=1 znamená Skip-gram, sg=0 znamená CBOW
model = Word2Vec(sentences=tokenized_sentences, 
                 vector_size=100,    # Dimenze vektorů
                 window=5,           # Velikost kontextového okna
                 min_count=1,        # Minimální počet výskytů slova
                 sg=1,               # Skip-gram model
                 workers=4)          # Počet vláken pro paralelní zpracování

# Uložení modelu
model.save("word2vec.model")

# Prozkoumání podobných slov
try:
    similar_to_king = model.wv.most_similar("král", topn=5)
    print("Slova podobná slovu 'král':")
    for word, score in similar_to_king:
        print(f"{word}: {score:.4f}")
except KeyError:
    print("Slovo 'král' není ve slovníku nebo nemá dostatek kontextu.")

# Vizualizace word vectors pomocí t-SNE
from sklearn.manifold import TSNE
import numpy as np

def visualize_embeddings(model, words=None):
    if words is None:
        words = list(model.wv.index_to_key)  # Všechna slova ve slovníku
    else:
        # Filtrování pouze těch slov, která jsou v modelu
        words = [word for word in words if word in model.wv]
    
    # Extrakce vektorů
    word_vectors = np.array([model.wv[word] for word in words])
    
    # Redukce dimenzionality pomocí t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(word_vectors)
    
    # Vizualizace
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.5)
    
    # Přidání popisků slov
    for i, word in enumerate(words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                     fontsize=9, alpha=0.7)
    
    plt.title('t-SNE vizualizace word embeddings')
    plt.grid(True)
    plt.show()

# Seznam slov pro vizualizaci (volitelně)
words_to_visualize = list(model.wv.index_to_key)[:20]  # Prvních 20 slov pro přehlednost
visualize_embeddings(model, words_to_visualize)
```

### Implementace Word2Vec pomocí TensorFlow/Keras

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Embedding, Dense, Lambda
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Příprava dat pro Skip-gram model
def generate_training_data(sentences, window_size, vocab_size):
    training_examples = []
    training_labels = []
    
    for sentence in sentences:
        for i, word in enumerate(sentence):
            # Definice kontextového okna
            window_start = max(0, i - window_size)
            window_end = min(len(sentence), i + window_size + 1)
            
            # Pro každé slovo v kontextovém okně (kromě cílového slova)
            for j in range(window_start, window_end):
                if j != i:  # Přeskočení cílového slova
                    training_examples.append(word)
                    training_labels.append(sentence[j])
    
    return np.array(training_examples), np.array(training_labels)

# Jednoduchá implementace Skip-gram modelu
def build_skip_gram_model(vocab_size, embedding_dim):
    model = Sequential([
        # Vstupní embedding vrstva
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1),
        # Zploštění výstupu embedding vrstvy
        Lambda(lambda x: tf.squeeze(x, axis=1)),
        # Výstupní vrstva s softmax aktivací
        Dense(vocab_size, activation='softmax')
    ])
    
    # Kompilace modelu
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

# Ukázka tréninku na malém datasetu
# Tokenizované věty (indexy slov)
word_to_id = {'král': 0, 'vládne': 1, 'v': 2, 'královském': 3, 'paláci': 4,
              'královna': 5, 'nosí': 6, 'krásnou': 7, 'korunu': 8}
id_to_word = {v: k for k, v in word_to_id.items()}

sentences_indexed = [
    [0, 1, 2, 3, 4],  # Král vládne v královském paláci
    [5, 6, 7, 8]      # Královna nosí krásnou korunu
]

# Parametry modelu
vocab_size = len(word_to_id)
embedding_dim = 5
window_size = 2

# Generování trénovacích dat
X, y = generate_training_data(sentences_indexed, window_size, vocab_size)

# Vytvoření modelu
model = build_skip_gram_model(vocab_size, embedding_dim)

# Trénování modelu
history = model.fit(X, y, epochs=100, verbose=0)

# Získání natrénovaných embeddingů
embeddings = model.layers[0].get_weights()[0]

# Vizualizace průběhu trénování
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()

# Výpis naučených embeddingů
print("Naučené embeddingy:")
for word, idx in word_to_id.items():
    print(f"{word}: {embeddings[idx]}")
```

## GloVe embeddings

GloVe (Global Vectors for Word Representation) je algoritmus pro učení word embeddings, který kombinuje lokální kontextové okno s globální statistikou spoluvýskytu slov v celém korpusu.

```python
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Příprava dat pro GloVe
def build_co_occurrence_matrix(sentences, window_size, vocab):
    vocab_size = len(vocab)
    word_to_id = {word: i for i, word in enumerate(vocab)}
    
    # Inicializace matice spoluvýskytu
    co_occurrence = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    
    # Naplnění matice spoluvýskytu
    for sentence in sentences:
        for i, center_word in enumerate(sentence):
            if center_word not in word_to_id:
                continue
                
            center_id = word_to_id[center_word]
            
            # Definice kontextového okna
            window_start = max(0, i - window_size)
            window_end = min(len(sentence), i + window_size + 1)
            
            # Počítání spoluvýskytů
            for j in range(window_start, window_end):
                if j != i and sentence[j] in word_to_id:
                    context_id = word_to_id[sentence[j]]
                    # Lze upravit váhu podle vzdálenosti
                    distance = abs(j - i)
                    weight = 1.0 / distance
                    co_occurrence[center_id, context_id] += weight
    
    return co_occurrence

# Načtení předtrénovaných GloVe embeddingů
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Příklad použití předtrénovaných GloVe embeddingů
def use_pretrained_glove():
    # Poznámka: Musíte si stáhnout předtrénované GloVe embeddingy z
    # https://nlp.stanford.edu/projects/glove/
    glove_file = "glove.6B.100d.txt"
    try:
        embeddings = load_glove_embeddings(glove_file)
        print(f"Načteno {len(embeddings)} GloVe embeddingů s dimenzí {len(next(iter(embeddings.values())))}")
        
        # Ukázka vektorových operací
        if 'king' in embeddings and 'man' in embeddings and 'woman' in embeddings:
            analogy_vector = embeddings['king'] - embeddings['man'] + embeddings['woman']
            
            # Hledání nejbližších vektorů
            similarities = {}
            for word, vec in embeddings.items():
                similarities[word] = np.dot(vec, analogy_vector) / (np.linalg.norm(vec) * np.linalg.norm(analogy_vector))
            
            # Výpis nejpodobnějších slov
            top_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
            print("Výsledek analogie 'king - man + woman':")
            for word, similarity in top_similar:
                print(f"{word}: {similarity:.4f}")
        else:
            print("Některá požadovaná slova nejsou v embeddingu.")
            
    except FileNotFoundError:
        print(f"Soubor {glove_file} nenalezen. Stáhněte předtrénované embeddingy.")

# Demo funkce pro vytvoření jednoduchých embeddingů
def demo_simple_glove():
    sentences = [
        ["pes", "má", "rád", "kosti"],
        ["kočka", "má", "ráda", "mléko"],
        ["pes", "honí", "kočku"],
        ["kočka", "honí", "myš"]
    ]
    
    vocab = sorted(list(set(word for sentence in sentences for word in sentence)))
    print(f"Slovník: {vocab}")
    
    # Vytvoření matice spoluvýskytu
    co_occurrence = build_co_occurrence_matrix(sentences, window_size=2, vocab=vocab)
    print("Matice spoluvýskytu:")
    pd.DataFrame(co_occurrence, index=vocab, columns=vocab)
    
    # V reálné implementaci GloVe bychom nyní použili tuto matici k učení embeddingů
    # pomocí faktorizace matice nebo optimalizačního algoritmu
    
    # Pro demonstrační účely použijeme SVD pro získání embeddingů
    U, S, Vh = np.linalg.svd(co_occurrence, full_matrices=False)
    embedding_dim = min(2, len(S))  # Pro vizualizaci použijeme 2 dimenze
    embeddings = U[:, :embedding_dim] * np.sqrt(S[:embedding_dim])
    
    # Vizualizace embeddingů
    plt.figure(figsize=(10, 6))
    plt.scatter(embeddings[:, 0], embeddings[:, 1] if embedding_dim > 1 else np.zeros_like(embeddings[:, 0]))
    
    for i, word in enumerate(vocab):
        plt.annotate(word, (embeddings[i, 0], embeddings[i, 1] if embedding_dim > 1 else 0))
    
    plt.title('Jednoduché GloVe embeddingy')
    plt.xlabel('Dimenze 1')
    plt.ylabel('Dimenze 2')
    plt.grid(True)
    plt.show()

# Spuštění demo funkcí
print("Demo jednoduchých GloVe embeddingů:")
demo_simple_glove()

print("\nUkázka použití předtrénovaných GloVe embeddingů:")
try:
    use_pretrained_glove()
except Exception as e:
    print(f"Chyba při načítání předtrénovaných embeddingů: {e}")
    print("Pro použití předtrénovaných embeddingů stáhněte soubory z https://nlp.stanford.edu/projects/glove/")
```

## FastText

FastText je rozšíření modelu Word2Vec, které bere v úvahu podslova (n-gramy znaků). To umožňuje lépe zachytit morfologické informace a generovat embeddingy i pro slova, která nebyla v trénovacích datech.

```python
import gensim
from gensim.models import FastText
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Příprava dat
sentences = [
    ["pes", "má", "rád", "kosti"],
    ["kočka", "má", "ráda", "mléko"],
    ["pes", "honí", "kočku"],
    ["kočka", "honí", "myš"],
    ["jablko", "je", "ovoce"],
    ["auto", "jede", "rychle"],
    ["autobus", "jede", "pomalu"]
]

# Trénování FastText modelu
model = FastText(
    sentences,
    vector_size=100,    # Dimenze vektorů
    window=3,           # Velikost kontextového okna
    min_count=1,        # Minimální počet výskytů slova
    workers=4,          # Počet vláken
    sg=1,               # Skip-gram model (0 pro CBOW)
    min_n=2,            # Minimální délka n-gramů znaků
    max_n=5             # Maximální délka n-gramů znaků
)

# Uložení modelu
model.save("fasttext.model")

# Získání word embeddingů
word_vectors = model.wv

# Ukázka embeddingů pro známá slova
print("Embeddingy pro známá slova:")
for word in ["pes", "kočka", "auto"]:
    if word in word_vectors:
        print(f"{word}: {word_vectors[word][:5]}...")  # Zobrazení prvních 5 hodnot

# Ukázka výhody FastTextu - embeddingy pro slova mimo trénovací data
print("\nEmbeddingy pro slova mimo trénovací data:")
for word in ["autíčko", "kočička", "pejsek"]:
    if word not in word_vectors:
        print(f"{word} není v trénovacím slovníku, ale FastText může vygenerovat embedding:")
        print(f"{word}: {word_vectors[word][:5]}...")  # Zobrazení prvních 5 hodnot

# Vizualizace podobnosti slov
def plot_word_similarities(model, reference_words, num_neighbors=5):
    """Vizualizace podobných slov pro každé referenční slovo"""
    plt.figure(figsize=(15, 10))
    
    for i, ref_word in enumerate(reference_words):
        if ref_word not in model.wv:
            print(f"Slovo '{ref_word}' není ve slovníku.")
            continue
            
        # Získání podobných slov
        similar_words = [word for word, _ in model.wv.most_similar(ref_word, topn=num_neighbors)]
        similar_words.insert(0, ref_word)
        
        # Extrakce vektorů
        vectors = np.array([model.wv[word] for word in similar_words])
        
        # PCA pro vizualizaci
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)
        
        # Vytvoření subplotu
        plt.subplot(2, len(reference_words), i + 1)
        plt.scatter(vectors_2d[1:, 0], vectors_2d[1:, 1], alpha=0.7)
        plt.scatter(vectors_2d[0, 0], vectors_2d[0, 1], color='red', s=100, alpha=0.7)
        
        # Přidání popisků
        for j, word in enumerate(similar_words):
            plt.annotate(word, (vectors_2d[j, 0], vectors_2d[j, 1]))
        
        plt.title(f"Slova podobná '{ref_word}'")
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Vizualizace podobnosti slov
plot_word_similarities(model, ["pes", "auto"])
```

## Kontextové embeddings

Kontextové embeddings, jako jsou BERT, ELMo nebo GPT, generují různé reprezentace pro stejné slovo v závislosti na jeho kontextu. Tyto modely zachycují kontextové informace a řeší problémy s polysémií (slova s více významy).

```python
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Načtení předtrénovaného BERT modelu a tokenizeru
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Nastavení modelu do režimu evaluace
model.eval()

# Funkce pro získání BERT embeddingů pro větu
def get_bert_embeddings(sentence, tokenizer, model):
    # Tokenizace a převod na tenzory
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    
    # Získání embeddingů z BERT
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Používáme poslední skrytou vrstvu modelu - všechny tokeny
    last_hidden_states = outputs.last_hidden_state
    
    # Převod na numpy pro další zpracování
    return inputs.input_ids, inputs.attention_mask, last_hidden_states.numpy()

# Demonstrace kontextových embeddingů na slově s různými významy
sentences = [
    "Banka poskytla výhodný úvěr pro nákup nemovitosti.",
    "Na břehu řeky byla písečná banka.",
    "Auto zaparkovalo u banky na náměstí.",
    "Krevní banka potřebuje dobrovolné dárce."
]

# Získání embeddingů pro každou větu
tokenized_sentences = []
embeddings = []

for sentence in sentences:
    tokens, mask, emb = get_bert_embeddings(sentence, tokenizer, model)
    tokenized_sentences.append(tokens)
    embeddings.append(emb)

# Vizualizace kontextových embeddingů pro slovo "banka"
def visualize_contextual_embeddings(sentences, tokenized_sentences, embeddings, target_word):
    plt.figure(figsize=(12, 8))
    
    word_embeddings = []
    contexts = []
    
    for i, (sentence, tokens, embedding) in enumerate(zip(sentences, tokenized_sentences, embeddings)):
        # Dekódování tokenů
        token_words = tokenizer.convert_ids_to_tokens(tokens[0])
        
        # Hledání cílového slova
        for j, token in enumerate(token_words):
            if target_word.lower() in token.lower():
                # Získání embeddigu pro token
                word_embedding = embedding[0, j, :]
                word_embeddings.append(word_embedding)
                contexts.append(sentence)
                print(f"Nalezeno slovo '{target_word}' v kontextu: '{sentence}'")
                break
    
    # Pokud jsme našli embeddingy pro cílové slovo
    if word_embeddings:
        # Převod na numpy array
        word_embeddings = np.array(word_embeddings)
        
        # Redukce dimenzí pro vizualizaci
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(word_embeddings)
        
        # Vizualizace
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=100, alpha=0.7)
        
        # Přidání popisků
        for i, context in enumerate(contexts):
            plt.annotate(f"Kontext {i+1}", (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), 
                         fontsize=9, alpha=0.8)
        
        plt.title(f'BERT embeddingy pro slovo "{target_word}" v různých kontextech')
        plt.grid(True)
        
        # Přidání legendy s kontexty
        for i, context in enumerate(contexts):
            plt.figtext(0.02, 0.95 - (i*0.05), f"Kontext {i+1}: {context}", fontsize=9)
            
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.show()
    else:
        print(f"Slovo '{target_word}' nebylo nalezeno v žádném kontextu.")

# Vizualizace kontextových embeddingů pro slovo "banka"
visualize_contextual_embeddings(sentences, tokenized_sentences, embeddings, "banka")
```

## Základy jazykových modelů

Jazykové modely jsou statistické modely nebo neuronové sítě, které se učí pravděpodobnostní distribuce nad sekvencemi slov nebo tokenů. Používají se pro predikci pravděpodobnosti výskytu určité posloupnosti slov.

**Cíl jazykových modelů**: Odhadnout P(w₁, w₂, ..., wₙ), tj. pravděpodobnost výskytu sekvence slov w₁, w₂, ..., wₙ.

Pomocí řetízkového pravidla lze tuto pravděpodobnost rozložit:  
P(w₁, w₂, ..., wₙ) = P(w₁) · P(w₂|w₁) · P(w₃|w₁,w₂) · ... · P(wₙ|w₁,...,wₙ₋₁)

```python
# Základní ilustrace principu jazykových modelů

# Ukázkový korpus
corpus = [
    "kočka sedí na okně",
    "pes běhá po zahradě",
    "kočka číhá na myš",
    "pes štěká na kočku",
    "myš se bojí kočky",
]

# Vytvoření n-gramů ze slov
def create_ngrams(sentence, n):
    words = sentence.split()
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

# Tvorba unigramů, bigramů a trigramů
unigrams = []
bigrams = []
trigrams = []

for sentence in corpus:
    unigrams.extend(create_ngrams(sentence, 1))
    bigrams.extend(create_ngrams(sentence, 2))
    if len(sentence.split()) >= 3:
        trigrams.extend(create_ngrams(sentence, 3))

# Výpočet frekvencí
def calculate_frequencies(ngrams):
    freq = {}
    for ngram in ngrams:
        if ngram in freq:
            freq[ngram] += 1
        else:
            freq[ngram] = 1
    return freq

unigram_freq = calculate_frequencies(unigrams)
bigram_freq = calculate_frequencies(bigrams)
trigram_freq = calculate_frequencies(trigrams)

# Výpis základních statistik
print(f"Počet unikátních unigramů: {len(unigram_freq)}")
print(f"Počet unikátních bigramů: {len(bigram_freq)}")
print(f"Počet unikátních trigramů: {len(trigram_freq)}")

# Top 5 nejčastějších unigramů a bigramů
sorted_unigrams = sorted(unigram_freq.items(), key=lambda x: x[1], reverse=True)
sorted_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)

print("\nTop 5 nejčastějších unigramů:")
for gram, freq in sorted_unigrams[:5]:
    print(f"'{gram}': {freq}")

print("\nTop 5 nejčastějších bigramů:")
for gram, freq in sorted_bigrams[:5]:
    print(f"'{gram}': {freq}")

# Výpočet pravděpodobnosti pro jednoduchý bigram model
def bigram_probability(bigram, unigram_freq, bigram_freq):
    """Vypočítá podmíněnou pravděpodobnost P(w2|w1) pro bigram 'w1 w2'"""
    words = bigram.split()
    if len(words) != 2:
        return 0
    
    w1, w2 = words
    
    # P(w2|w1) = počet(w1 w2) / počet(w1)
    if w1 not in unigram_freq or bigram not in bigram_freq:
        return 0
    
    return bigram_freq[bigram] / unigram_freq[w1]

# Příklad výpočtu pravděpodobností
print("\nPříklady podmíněných pravděpodobností:")
example_bigrams = ["kočka sedí", "kočka číhá", "pes štěká"]
for bigram in example_bigrams:
    prob = bigram_probability(bigram, unigram_freq, bigram_freq)
    print(f"P({bigram.split()[1]}|{bigram.split()[0]}) = {prob:.2f}")
```

## N-gram jazykové modely

N-gramové jazykové modely předpovídají následující slovo na základě předchozích n-1 slov. Jsou založeny na Markovově předpokladu, že pravděpodobnost slova závisí pouze na předchozích n-1 slovech.

```python
import numpy as np
import random
from collections import defaultdict, Counter

# Korpus pro trénování
corpus = [
    "jak se dnes máš",
    "mám se dobře děkuji",
    "jak se dnes cítíš",
    "cítím se skvěle",
    "máš dnes hodně práce",
    "mám hodně práce a málo času",
    "jak se jmenuješ",
    "jmenuji se petr",
    "máš rád programování",
    "programování je zábavné"
]

# Tokenizace - rozdělení vět na slova
tokenized_corpus = [sentence.split() for sentence in corpus]

# Přidání značek pro začátek a konec věty
def add_sentence_tokens(sentences):
    return [["<s>"] + sentence + ["</s>"] for sentence in sentences]

tokenized_corpus = add_sentence_tokens(tokenized_corpus)

# Vytvoření n-gramů
def create_ngrams(sentences, n):
    ngrams = []
    for sentence in sentences:
        # Generování n-gramů pro každou větu
        for i in range(len(sentence) - n + 1):
            ngrams.append(tuple(sentence[i:i+n]))
    return ngrams

# Vytvoření bigramů a trigramů
bigrams = create_ngrams(tokenized_corpus, 2)
trigrams = create_ngrams(tokenized_corpus, 3)

# Vytvoření modelu
class NGramLanguageModel:
    def __init__(self, n):
        self.n = n  # Stupeň n-gramu
        self.model = defaultdict(Counter)
        self.vocab = set()
        
    def train(self, sentences):
        # Trénování n-gramového modelu
        for sentence in sentences:
            for i in range(len(sentence) - self.n + 1):
                prefix = tuple(sentence[i:i + self.n - 1])
                suffix = sentence[i + self.n - 1]
                self.model[prefix][suffix] += 1
                self.vocab.add(suffix)
        
        # Přidání všech slov do slovníku
        for sentence in sentences:
            for word in sentence:
                self.vocab.add(word)
    
    def predict_next_word(self, context, smoothing=0.1):
        """Předpovídá další slovo na základě kontextu s Laplacovým vyhlazováním"""
        context_tuple = tuple(context[-(self.n-1):])  # Použití pouze posledních n-1 slov
        
        if context_tuple in self.model:
            counter = self.model[context_tuple]
            total = sum(counter.values())
            
            # Aplikace vyhlazování
            probabilities = {}
            for word in self.vocab:
                # Přidání vyhlazování: (počet + alpha) / (celkem + alpha * velikost_slovníku)
                probabilities[word] = (counter[word] + smoothing) / (total + smoothing * len(self.vocab))
            
            return probabilities
        else:
            # Pokud kontext neexistuje, použijeme uniformní distribuci
            return {word: 1/len(self.vocab) for word in self.vocab}
    
    def generate_sentence(self, start=None, max_length=20):
        """Generuje větu s pomocí modelu"""
        if start is None:
            context = ["<s>"] * (self.n - 1)  # Začátek věty
        else:
            # Ověření délky počátečního kontextu
            if len(start) >= self.n - 1:
                context = start[-(self.n-1):]
            else:
                context = ["<s>"] * (self.n - 1 - len(start)) + start
        
        sentence = list(context)
        
        # Generování nových slov
        for _ in range(max_length):
            probabilities = self.predict_next_word(sentence)
            
            # Výběr slova podle pravděpodobnosti
            words, probs = zip(*probabilities.items())
            next_word = np.random.choice(words, p=probs)
            
            sentence.append(next_word)
            
            # Kontrola ukončení věty
            if next_word == "</s>":
                break
        
        # Odstranění značek začátku a konce věty
        return [word for word in sentence if word not in ["<s>", "</s>"]]

# Vytvoření a trénování modelů
bigram_model = NGramLanguageModel(n=2)
trigram_model = NGramLanguageModel(n=3)

bigram_model.train(tokenized_corpus)
trigram_model.train(tokenized_corpus)

# Generování vět
print("Bigramový model - vygenerované věty:")
for _ in range(3):
    sentence = bigram_model.generate_sentence()
    print(" ".join(sentence))

print("\nTrigramový model - vygenerované věty:")
for _ in range(3):
    sentence = trigram_model.generate_sentence()
    print(" ".join(sentence))

# Ukázka predikce následujícího slova
test_contexts = [["jak", "se"], ["mám", "hodně"], ["programování", "je"]]

print("\nPředpověď následujícího slova:")
for context in test_contexts:
    print(f"\nKontext: '{' '.join(context)}'")
    
    # Získání top 3 nejpravděpodobnějších slov z bigramového modelu
    if len(context) >= 1:
        bigram_probs = bigram_model.predict_next_word([context[-1]])
        top_bigram_words = sorted(bigram_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print("Bigramový model - top 3 predikce:")
        for word, prob in top_bigram_words:
            print(f"  '{word}': {prob:.4f}")
    
    # Získání top 3 nejpravděpodobnějších slov z trigramového modelu
    if len(context) >= 2:
        trigram_probs = trigram_model.predict_next_word(context)
        top_trigram_words = sorted(trigram_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print("Trigramový model - top 3 predikce:")
        for word, prob in top_trigram_words:
            print(f"  '{word}': {prob:.4f}")
```

## Neuronové jazykové modely

Neuronové jazykové modely používají neuronové sítě pro modelování pravděpodobnostní distribuce slov. Na rozdíl od n-gramových modelů dokáží zachytit dlouhodobé závislosti a generalizovat i na neviděné kombinace slov.

### Jednoduchý RNN jazykový model

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Korpus vět
sentences = [
    "jak se dnes máš",
    "mám se dobře děkuji",
    "jak se dnes cítíš",
    "cítím se skvěle",
    "máš dnes hodně práce",
    "mám hodně práce a málo času",
    "jak se jmenuješ",
    "jmenuji se petr",
    "máš rád programování",
    "programování je zábavné"
]

# Tokenizace
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1
print(f"Celkem unikátních slov: {total_words-1}")

# Převod vět na sekvence tokenů
sequences = tokenizer.texts_to_sequences(sentences)

# Vytvoření trénovacích dat
# Pro každou pozici v sekvenci vytvoříme vstup (předchozí slova) a výstup (následující slovo)
input_sequences = []
output_labels = []

for sequence in sequences:
    for i in range(1, len(sequence)):
        # Vstup: slova od začátku až po pozici i
        input_sequences.append(sequence[:i])
        # Výstup: slovo na pozici i
        output_labels.append(sequence[i])

# Padding sekvencí na stejnou délku
max_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

# Převod výstupních labelů na kategoriální formát (one-hot encoding)
output_labels = tf.keras.utils.to_categorical(output_labels, num_classes=total_words)

# Definice RNN jazykového modelu
embedding_dim = 32  # Dimenze word embeddingů
hidden_units = 64   # Počet skrytých jednotek RNN

model = Sequential([
    Embedding(input_dim=total_words, output_dim=embedding_dim, input_length=max_len),
    SimpleRNN(hidden_units),
    Dense(total_words, activation='softmax')  # Predikce pravděpodobností pro každé slovo ve slovníku
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Trénování modelu
history = model.fit(
    input_sequences,
    output_labels,
    epochs=100,
    verbose=1,
    batch_size=4
)

# Vizualizace výsledků trénování
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()

# Funkce pro generování textu pomocí natrénovaného modelu
def generate_text(seed_text, next_words, model, tokenizer, max_len):
    """Generuje text pomocí natrénovaného RNN modelu"""
    for _ in range(next_words):
        # Tokenizace seed textu
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # Padding sekvence
        token_list = pad_sequences([token_list], maxlen=max_len, padding='pre')
        # Predikce následujícího tokenu
        probabilities = model.predict(token_list)[0]
        predicted = np.argmax(probabilities)
        
        # Převod tokenu zpět na slovo
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        # Přidání nového slova do seed textu
        seed_text += " " + output_word
    
    return seed_text

# Ukázka generování textu
seed_texts = ["jak se", "mám hodně", "programování"]
for seed in seed_texts:
    generated_text = generate_text(seed, next_words=3, model=model, tokenizer=tokenizer, max_len=max_len)
    print(f"Seed text: '{seed}'")
    print(f"Generated text: '{generated_text}'")
    print()
```

### Model s LSTM

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Použijeme stejný korpus jako v předchozím příkladu
sentences = [
    "jak se dnes máš",
    "mám se dobře děkuji",
    "jak se dnes cítíš",
    "cítím se skvěle",
    "máš dnes hodně práce",
    "mám hodně práce a málo času",
    "jak se jmenuješ",
    "jmenuji se petr",
    "máš rád programování",
    "programování je zábavné"
]

# Tokenizace
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1
print(f"Celkem unikátních slov: {total_words-1}")

# Převod vět na sekvence tokenů
sequences = tokenizer.texts_to_sequences(sentences)

# Vytvoření trénovacích dat
input_sequences = []
output_labels = []

for sequence in sequences:
    for i in range(1, len(sequence)):
        input_sequences.append(sequence[:i])
        output_labels.append(sequence[i])

max_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')
output_labels = tf.keras.utils.to_categorical(output_labels, num_classes=total_words)

# Definice LSTM jazykového modelu
embedding_dim = 32
hidden_units = 64

model = Sequential([
    Embedding(input_dim=total_words, output_dim=embedding_dim, input_length=max_len),
    LSTM(hidden_units, return_sequences=True),  # Vrátí výstupy pro všechny časové kroky
    Dropout(0.2),                              # Dropout pro prevenci přetrénování
    LSTM(hidden_units),                        # Druhá LSTM vrstva
    Dropout(0.2),
    Dense(total_words, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Trénování modelu
history = model.fit(
    input_sequences,
    output_labels,
    epochs=150,
    verbose=1,
    batch_size=4
)

# Vizualizace výsledků trénování
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()

# Funkce pro generování textu pomocí natrénovaného modelu
def generate_text(seed_text, next_words, model, tokenizer, max_len):
    """Generuje text pomocí natrénovaného LSTM modelu"""
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len, padding='pre')
        probabilities = model.predict(token_list)[0]
        
        # Top-k sampling: Výběr z top k nejpravděpodobnějších slov
        k = 3
        top_indices = np.argsort(probabilities)[-k:]
        top_probs = probabilities[top_indices]
        top_probs = top_probs / np.sum(top_probs)  # Normalizace pravděpodobností
        
        # Náhodný výběr z top-k slov podle jejich pravděpodobností
        predicted = np.random.choice(top_indices, p=top_probs)
        
        # Převod tokenu zpět na slovo
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        seed_text += " " + output_word
    
    return seed_text

# Ukázka generování textu s LSTM modelem
seed_texts = ["jak se", "mám hodně", "programování"]
for seed in seed_texts:
    generated_text = generate_text(seed, next_words=5, model=model, tokenizer=tokenizer, max_len=max_len)
    print(f"Seed text: '{seed}'")
    print(f"Generated text: '{generated_text}'")
    print()
```

## Implementace vlastního jazykového modelu

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
import re

# Příprava dat
def load_simple_dataset():
    """Načte jednoduchý dataset pro demonstraci"""
    # Můžete nahradit vlastními daty
    texts = [
        "Python je vysokoúrovňový programovací jazyk.",
        "Python byl vytvořen Guido van Rossumem v roce 1991.",
        "Python je populární pro strojové učení.",
        "Strojové učení je podoblast umělé inteligence.",
        "Umělá inteligence se rychle vyvíjí.",
        "Python má jednoduchou a čitelnou syntaxi.",
        "Programování v Pythonu je zábavné.",
        "Knihovny jako TensorFlow a PyTorch jsou často používány pro strojové učení.",
        "Neuronové sítě jsou základem hlubokého učení.",
        "Python je interpretovaný jazyk.",
        "Datová věda využívá Pythonu pro analýzu dat.",
        "Natural Language Processing zpracovává přirozený jazyk."
    ]
    return texts

# Načtení a předzpracování dat
texts = load_simple_dataset()

# Jednoduchá tokenizace a čištění textu
def preprocess_texts(texts):
    preprocessed_texts = []
    for text in texts:
        # Převod na malá písmena
        text = text.lower()
        # Odstranění interpunkce
        text = re.sub(r'[^\w\s]', '', text)
        preprocessed_texts.append(text)
    return preprocessed_texts

preprocessed_texts = preprocess_texts(texts)

# Tokenizace slov
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_texts)
total_words = len(tokenizer.word_index) + 1

print(f"Velikost slovníku: {total_words-1} slov")
print(f"Ukázka slovníku: {list(tokenizer.word_index.items())[:5]}")

# Vytvoření sekvencí pro trénování
sequences = []
for text in preprocessed_texts:
    token_list = tokenizer.texts_to_sequences([text])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        sequences.append(n_gram_sequence)

print(f"Celkem sekvencí: {len(sequences)}")
if sequences:
    print(f"Ukázka sekvence: {sequences[0]}")

# Příprava dat pro trénink
max_sequence_len = max([len(seq) for seq in sequences])
input_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')

# Rozdělení na vstupy a výstupy
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

print(f"Tvar vstupních dat: {X.shape}")
print(f"Tvar výstupních dat: {y.shape}")

# Definice vlastního jazykového modelu
def build_language_model(vocab_size, embedding_dim, rnn_units, dropout_rate=0.2):
    """
    Vytvoří jazykový model s LSTM vrstvami
    
    Args:
        vocab_size: Velikost slovníku
        embedding_dim: Dimenze embedding vrstvy
        rnn_units: Počet jednotek v LSTM vrstvách
        dropout_rate: Míra dropoutu pro prevenci přetrénování
    
    Returns:
        Model: Zkompilovaný Keras model
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_sequence_len-1),
        LSTM(rnn_units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(rnn_units),
        Dropout(dropout_rate),
        Dense(vocab_size//2, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

# Vytvoření modelu
embedding_dim = 64
rnn_units = 128
dropout_rate = 0.3

model = build_language_model(total_words, embedding_dim, rnn_units, dropout_rate)
model.summary()

# Callbacky pro trénink
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

early_stopping = EarlyStopping(monitor='loss', patience=10)

# Trénování modelu
history = model.fit(
    X, y,
    epochs=100,
    batch_size=4,
    callbacks=[checkpoint_callback, early_stopping],
    verbose=1
)

# Vizualizace průběhu trénování
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Přesnost modelu')
plt.ylabel('Přesnost')
plt.xlabel('Epocha')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Ztráta modelu')
plt.ylabel('Ztráta')
plt.xlabel('Epocha')
plt.tight_layout()
plt.show()

# Funkce pro generování textu
def generate_text(model, tokenizer, seed_text, next_words, max_sequence_len, temperature=1.0):
    """
    Generuje text na základě natrénovaného modelu
    
    Args:
        model: Natrénovaný jazykový model
        tokenizer: Tokenizer použitý při přípravě dat
        seed_text: Počáteční text pro generování
        next_words: Počet slov k vygenerování
        max_sequence_len: Maximální délka vstupní sekvence
        temperature: Teplota pro sampling (vyšší hodnota = více náhodnosti)
    
    Returns:
        str: Vygenerovaný text
    """
    # Předzpracování seed textu
    seed_text = seed_text.lower()
    seed_text = re.sub(r'[^\w\s]', '', seed_text)
    
    # Generování textu
    for _ in range(next_words):
        # Tokenizace seed textu
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # Padding sekvence
        padded_sequence = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # Predikce
        predicted_probs = model.predict(padded_sequence, verbose=0)[0]
        
        # Aplikace temperature
        predicted_probs = np.log(predicted_probs) / temperature
        exp_probs = np.exp(predicted_probs)
        predicted_probs = exp_probs / np.sum(exp_probs)
        
        # Sampling z distribuce pravděpodobnosti
        predicted_id = np.random.choice(len(predicted_probs), p=predicted_probs)
        
        # Převod ID na slovo
        predicted_word = ""
        for word, idx in tokenizer.word_index.items():
            if idx == predicted_id:
                predicted_word = word
                break
        
        # Přidání predikovaného slova k seed textu
        seed_text += " " + predicted_word
    
    return seed_text

# Ukázka generování textu
seed_texts = [
    "python je",
    "strojové učení",
    "umělá inteligence"
]

for seed in seed_texts:
    for temp in [0.5, 1.0]:
        generated_text = generate_text(
            model,
            tokenizer,
            seed,
            next_words=10,
            max_sequence_len=max_sequence_len,
            temperature=temp
        )
        print(f"Seed: '{seed}', Temperature: {temp}")
        print(f"Generated: '{generated_text}'")
        print()
```

## Praktické aplikace

Word embeddings a jazykové modely mají širokou škálu praktických aplikací v oblasti zpracování přirozeného jazyka a strojového učení.

### Klasifikace textu

Vektorové reprezentace slov jsou velmi užitečné pro klasifikační úlohy jako sentiment analýza, detekce spamu nebo kategorizace článků.

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Příklad dat pro sentiment analýzu
texts = [
    "Film se mi velmi líbil, herci odvedli skvělou práci.",
    "Naprosto úžasná kniha, nemohl jsem přestat číst.",
    "Restaurace byla výborná, jídlo bylo chutné a obsluha milá.",
    "Ten produkt mě zklamal, nefunguje jak by měl.",
    "Hrozný zážitek, nikdy více.",
    "Ztráta času a peněz, nedoporučuji.",
    "Průměrný film, nic mimořádného.",
    "Byl to celkem dobrý koncert, ale čekal jsem víc."
]

# Přiřazení sentimentu: 1 = pozitivní, 0 = neutrální, -1 = negativní
sentiments = [1, 1, 1, -1, -1, -1, 0, 0]

# Tokenizace a příprava dat
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding sekvencí na jednotnou délku
max_length = 20
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Převod labelů na kategorie
labels = np.array(sentiments) + 1  # Posun z (-1, 0, 1) na (0, 1, 2)

# Rozdělení na trénovací a testovací data
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)

# Vytvoření modelu
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 16

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(32),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 kategorie: negativní, neutrální, pozitivní
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Trénování modelu
history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    verbose=0
)

# Vizualizace výsledků
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trénovací přesnost')
plt.plot(history.history['val_accuracy'], label='Validační přesnost')
plt.title('Přesnost modelu')
plt.xlabel('Epocha')
plt.ylabel('Přesnost')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trénovací loss')
plt.plot(history.history['val_loss'], label='Validační loss')
plt.title('Loss modelu')
plt.xlabel('Epocha')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Testování modelu na nových datech
new_texts = [
    "Ten film byl naprosto úžasný!",
    "Nemůžu to doporučit, naprostá katastrofa.",
    "Film byl docela slušný, ale nic víc."
]

# Příprava nových dat
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=max_length, padding='post')

# Predikce
predictions = model.predict(new_padded)
pred_classes = np.argmax(predictions, axis=1)

# Převod zpět na sentimenty
sentiment_map = {0: "negativní", 1: "neutrální", 2: "pozitivní"}
for i, text in enumerate(new_texts):
    print(f"Text: '{text}'")
    print(f"Předpovězený sentiment: {sentiment_map[pred_classes[i]]}")
    print(f"Jistota: {predictions[i][pred_classes[i]]:.4f}")
    print()
```

### Strojový překlad

Jazykové modely jsou základem moderních systémů strojového překladu.

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

# Jednoduchá ukázka strojového překladu (čeština -> angličtina)
# V reálné aplikaci byste použili mnohem větší dataset

cs_sentences = [
    "ahoj jak se máš",
    "dobrý den",
    "děkuji mnohokrát",
    "na shledanou",
    "mám se dobře",
    "kolik je hodin",
    "kde je nejbližší restaurace"
]

en_sentences = [
    "hello how are you",
    "good day",
    "thank you very much",
    "goodbye",
    "i am fine",
    "what time is it",
    "where is the nearest restaurant"
]

# Tokenizace pro češtinu
cs_tokenizer = Tokenizer()
cs_tokenizer.fit_on_texts(cs_sentences)
cs_sequences = cs_tokenizer.texts_to_sequences(cs_sentences)

# Tokenizace pro angličtinu
en_tokenizer = Tokenizer()
en_tokenizer.fit_on_texts(en_sentences)
en_sequences = en_tokenizer.texts_to_sequences(en_sentences)

# Přidání start/end tokenů pro angličtinu
en_tokenizer.word_index['<start>'] = len(en_tokenizer.word_index) + 1
en_tokenizer.word_index['<end>'] = len(en_tokenizer.word_index) + 1
en_tokenizer.index_word = {v: k for k, v in en_tokenizer.word_index.items()}

# Padding sekvencí
cs_max_len = max([len(seq) for seq in cs_sequences])
en_max_len = max([len(seq) for seq in en_sequences]) + 2  # +2 pro start/end tokeny

cs_padded = pad_sequences(cs_sequences, maxlen=cs_max_len, padding='post')

# Vytvoření sekvencí s tokeny start/end pro decoder
decoder_input_data = []
decoder_output_data = []
for seq in en_sequences:
    # Sekvence pro vstup decoderu: <start> + věta
    decoder_input = [en_tokenizer.word_index['<start>']] + seq
    # Sekvence pro výstup decoderu: věta + <end>
    decoder_output = seq + [en_tokenizer.word_index['<end>']]
    decoder_input_data.append(decoder_input)
    decoder_output_data.append(decoder_output)

decoder_input_padded = pad_sequences(decoder_input_data, maxlen=en_max_len, padding='post')
decoder_output_padded = pad_sequences(decoder_output_data, maxlen=en_max_len, padding='post')

# One-hot kódování výstupů pro training
en_vocab_size = len(en_tokenizer.word_index) + 1
decoder_output_onehot = np.zeros(
    (len(decoder_output_padded), en_max_len, en_vocab_size), 
    dtype="float32"
)

for i, seq in enumerate(decoder_output_padded):
    for j, token in enumerate(seq):
        if token > 0:  # Ignorujeme padding (0)
            decoder_output_onehot[i, j, token] = 1.

# Definice modelu enkodér-dekodér
# Enkodér
encoder_inputs = Input(shape=(cs_max_len,))
encoder_embedding = Embedding(len(cs_tokenizer.word_index) + 1, 128)(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Dekodér
decoder_inputs = Input(shape=(en_max_len,))
decoder_embedding = Embedding(len(en_tokenizer.word_index) + 1, 128)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(len(en_tokenizer.word_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model pro trénink
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# V reálné aplikaci byste model trénovali na mnohem větším datasetu,
# proto tuto část přeskočíme a ukážeme jen inferenční model

# Inferenční model - enkodér
encoder_model = Model(encoder_inputs, encoder_states)

# Inferenční model - dekodér
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# Funkce pro překlad
def translate_sentence(input_sentence):
    # Tokenizace a padding vstupní věty
    input_seq = cs_tokenizer.texts_to_sequences([input_sentence])
    input_padded = pad_sequences(input_seq, maxlen=cs_max_len, padding='post')
    
    # Enkódování vstupní sekvence
    states_value = encoder_model.predict(input_padded)
    
    # Generování výstupu pomocí dekodéru
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = en_tokenizer.word_index['<start>']
    
    stop_condition = False
    decoded_sentence = []
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        # Vzít token s nejvyšší pravděpodobností
        sampled_token_index = np.argmax(output_tokens[0, 0, :])
        if sampled_token_index == 0:  # Padding token
            break
            
        # Získání slova z indexu
        sampled_word = en_tokenizer.index_word.get(sampled_token_index, '<unk>')
        
        # Ukončení, pokud narazíme na konec nebo dosáhneme max délky
        if sampled_word == '<end>' or len(decoded_sentence) > en_max_len:
            stop_condition = True
        else:
            decoded_sentence.append(sampled_word)
        
        # Aktualizace cíle pro příští iteraci
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        # Aktualizace stavů
        states_value = [h, c]
    
    return ' '.join(decoded_sentence)

# Pro plnohodnotný překlad byste potřebovali natrénovaný model
print("Poznámka: Pro funkční překlad je třeba natrénovat model na větším datasetu.")
```

### Automatické doplňování textu

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

# Jednoduchý dataset pro ukázku
texts = [
    "Programování v Pythonu je velmi populární.",
    "Python je interpretovaný vysokoúrovňový programovací jazyk.",
    "Umělá inteligence se rychle rozvíjí.",
    "Strojové učení využívá algoritmy k analýze dat.",
    "Hluboké učení je podmnožina strojového učení.",
    "Neuronové sítě jsou základem hlubokého učení.",
    "Big data vyžadují speciální přístupy ke zpracování.",
    "Programátoři často používají Git pro správu verzí kódu.",
    "Cloud computing umožňuje škálovatelné nasazení aplikací.",
    "Internet věcí propojuje fyzické objekty s internetem."
]

# Tokenizace
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1

# Vytvoření sekvencí pro jazykový model
input_sequences = []
for text in texts:
    token_list = tokenizer.texts_to_sequences([text])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Padding sekvencí
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Vytvoření X (input) a y (target)
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Definice modelu pro automatické doplňování
model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len-1),
    Bidirectional(LSTM(150, return_sequences=True)),
    Bidirectional(LSTM(100)),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# V reálném případě byste model natrénovali na větším datasetu
# model.fit(X, y, epochs=100, batch_size=32)

# Funkce pro generování doplnění textu
def complete_text(seed_text, next_words=5):
    for _ in range(next_words):
        # Tokenizace vstupního textu
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # Padding
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # Predikce dalšího slova
        predicted = np.argmax(model.predict(token_list), axis=-1)
        
        # Konverze ID na slovo
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        # Přidání slova do seed_text
        seed_text += " " + output_word
    
    return seed_text

# Příklady doplnění textu (vyžadují natrénovaný model)
input_texts = [
    "Programování v Pythonu",
    "Umělá inteligence",
    "Neuronové sítě"
]

print("Poznámka: Pro funkční doplňování textu je třeba natrénovat model.")
```

### Analýza sentimentu

Word embeddings jsou velmi užitečné pro klasifikační úlohy jako sentiment analýza.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Načtení IMDB datasetu pro sentiment analýzu
num_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

# Padding sekvencí na jednotnou délku
maxlen = 200
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Vytvoření jednoduchého modelu s embedding vrstvou
embedding_dim = 16
model = Sequential([
    Embedding(num_words, embedding_dim, input_length=maxlen),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trénování modelu
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# Vyhodnocení modelu
results = model.evaluate(X_test, y_test)
print(f"Test accuracy: {results[1]:.4f}")

# Vizualizace průběhu trénování
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()

# Funkce pro získání slovníku z IMDB datasetu
def get_word_index():
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    reverse_word_index = {v: k for k, v in word_index.items()}
    return word_index, reverse_word_index

# Funkce pro převod sekvence tokenů zpět na text
def decode_review(sequence):
    _, reverse_word_index = get_word_index()
    return ' '.join([reverse_word_index.get(i, '?') for i in sequence])

# Ukázka predikce sentimentu
def predict_sentiment(text, model, word_index, maxlen):
    # Převod textu na sekvenci tokenů
    words = text.lower().split()
    tokens = []
    for word in words:
        if word in word_index:
            tokens.append(word_index[word])
        else:
            tokens.append(word_index["<UNK>"])
    
    # Padding
    if len(tokens) < maxlen:
        tokens = [0] * (maxlen - len(tokens)) + tokens
    else:
        tokens = tokens[-maxlen:]
    
    # Predikce
    prediction = model.predict(np.array([tokens]))[0][0]
    sentiment = "pozitivní" if prediction > 0.5 else "negativní"
    
    return sentiment, prediction

# Příklad recenze
sample_review = X_test[0]
word_index, _ = get_word_index()

print("Ukázka recenze:")
print(decode_review(sample_review))
print("\nSkutečný sentiment:", "pozitivní" if y_test[0] == 1 else "negativní")

# Predikce sentimentu by vyžadovala zpětné mapování textu na ID tokeny
# Pro jednoduchost zde tuto část přeskočíme
```

## Shrnutí a nejlepší postupy

### Klíčové body

1. **Word embeddings** jsou vektorové reprezentace slov, které zachycují sémantické a syntaktické vztahy mezi slovy v mnohorozměrném prostoru.

2. **Typy word embeddings**:
   - **Statické** (Word2Vec, GloVe, FastText): poskytují jeden vektor pro každé slovo.
   - **Kontextové** (ELMo, BERT): poskytují různé vektory pro stejné slovo v závislosti na kontextu.

3. **Jazykové modely** předpovídají pravděpodobnost výskytu sekvence slov nebo následujícího slova v sekvenci.
   - **n-gramové modely**: založené na četnosti sekvencí slov v korpusu.
   - **Neuronové jazykové modely**: využívají neuronové sítě pro modelování pravděpodobnostních distribucí slov.

4. **Výhody word embeddings**:
   - Zachycení sémantických vztahů mezi slovy
   - Snížení dimenzionality oproti one-hot encoding
   - Možnost provádět algebraické operace se slovy (např. "král" - "muž" + "žena" ≈ "královna")

5. **Moderní přístupy** využívají pokročilé architektury jako transformery, které výrazně zlepšily výsledky v oblasti NLP.

### Doporučené postupy

1. **Výběr vhodných word embeddings:**
   - Pro jednoduché úlohy a malé datasety jsou vhodné statické embeddings (Word2Vec, GloVe)
   - Pro složitější úlohy vyžadující porozumění kontextu jsou lepší kontextové embeddings (BERT)

```python
# Příklad použití předtrénovaných embeddings vs. vlastních
import gensim.downloader as api

# Předtrénované embeddings (výhoda: naučené na velkém korpusu)
word_vectors = api.load('word2vec-google-news-300')

# Vlastní embeddings (výhoda: specifické pro vaši doménu)
from gensim.models import Word2Vec
model = Word2Vec(sentences=your_sentences, vector_size=100, window=5, min_count=5)
```

2. **Příprava dat pro jazykové modely:**
   - Důkladné předzpracování textu (tokenizace, čištění, lemmatizace)
   - Správné řešení neznámých slov (OOV - out-of-vocabulary)
   - Augmentace dat pro lepší generalizaci

```python
# Příklad předzpracování textu
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_text(text):
    # Převod na malá písmena
    text = text.lower()
    
    # Odstranění speciálních znaků a číslic
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenizace
    tokens = word_tokenize(text)
    
    # Odstranění stop slov (volitelné)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatizace
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens
```

3. **Hyperparametry a architektura modelů:**
   - Volba správné dimenze word embeddings (100-300 je obvyklé)
   - Hloubka a šířka neuronových sítí dle složitosti úlohy
   - Použití regularizačních technik proti přetrénování

```python
# Příklad modelu s různými regularizačními technikami
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2

def create_model(vocab_size, embedding_dim, max_length):
    model = Sequential([
        # Embedding vrstva s L2 regularizací
        Embedding(
            vocab_size, 
            embedding_dim, 
            input_length=max_length,
            embeddings_regularizer=l2(1e-5)
        ),
        
        # LSTM vrstva s dropout a recurrent dropout
        LSTM(
            128, 
            dropout=0.2,               # Dropout na vstupech
            recurrent_dropout=0.2      # Dropout na rekurentních spojeních
        ),
        
        # Hustě propojené vrstvy s dropout
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.3),
        Dense(vocab_size, activation='softmax')
    ])
    
    return model
```

4. **Transfer learning:**
   - Využití předtrénovaných modelů místo trénování od nuly
   - Fine-tuning pro specifické úlohy

```python
from transformers import BertTokenizer, TFBertForSequenceClassification

# Načtení předtrénovaného BERT modelu
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Fine-tuning na vlastní úloze
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# model.fit(train_dataset, validation_data=val_dataset, epochs=3)
```

5. **Evaluace a testování:**
   - Správný výběr metrik podle typu úlohy
   - Cross-validace pro robustnější výsledky
   - Analýza chyb pro identifikaci slabých míst modelu

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X, y):
    # Predikce
    y_pred = model.predict(X)
    
    # Základní metriky
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Cross-validace
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.4f}")
```

### Porovnání různých přístupů

| Metoda | Výhody | Nevýhody | Ideální použití |
|--------|--------|----------|----------------|
| Word2Vec | Rychlý, efektivní, zachytí sémantiku | Statické embeddings, nezohledňuje kontext | Základní NLP úlohy, malé až střední projekty |
| GloVe | Kombinuje lokální a globální kontext | Statické embeddings | Úlohy vyžadující globální statistiky korpusu |
| FastText | Dokáže generovat vektory pro neviděná slova | Větší paměťová náročnost | Jazyky s bohatou morfologií |
| BERT/kontextové embeddings | Zachytí různé významy slov v kontextu | Výpočetně náročné | Komplexní úlohy vyžadující porozumění kontextu |
| n-gram modely | Jednoduché, interpretovatelné | Nezachytí dlouhodobé závislosti | Baseline modely, funkce doplňování |
| RNN/LSTM/GRU modely | Zachytí sekvence a dlouhodobé závislosti | Pomalejší trénování | Sekvenční úlohy, generování textu |
| Transformer modely | State-of-the-art výsledky v NLP | Vysoké výpočetní nároky | Komplexní NLP úlohy, generativní AI |

Word embeddings a jazykové modely představují základ pro mnoho moderních aplikací zpracování přirozeného jazyka. Správný výběr metod a modelů závisí na konkrétních požadavcích projektu, dostupných datech a výpočetních zdrojích.