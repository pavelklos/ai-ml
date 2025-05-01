# Rekurentní neuronové sítě (RNN) a LSTM sítě

## Obsah
1. Úvod do rekurentních neuronových sítí
2. Základní architektura RNN
3. Problém mizejícího a explodujícího gradientu
4. LSTM: Long Short-Term Memory sítě
5. Implementace RNN a LSTM v TensorFlow/Keras
6. Implementace RNN a LSTM v PyTorch
7. Praktické aplikace a příklady
8. Shrnutí a nejlepší postupy

## Úvod do rekurentních neuronových sítí

Rekurentní neuronové sítě (RNN) jsou specializovaným typem neuronových sítí navržených pro zpracování sekvenčních dat a dat s časovou závislostí. Na rozdíl od feedforward sítí (jako CNN) mají RNN vnitřní paměť - jejich výstup závisí nejen na aktuálním vstupu, ale i na předchozích vstupech v sekvenci.

**Klíčové vlastnosti RNN:**
- Vnitřní paměťové stavy umožňující zpracování sekvenčních dat
- Schopnost modelovat závislosti v čase
- Sdílení parametrů napříč časovými kroky
- Vhodné pro data různé délky (text, audio, časové řady)

## Základní architektura RNN

Základní RNN buňka přijímá vstupní data a předchozí skrytý stav a generuje nový skrytý stav a výstup.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense, Input
from tensorflow.keras.models import Model

# Ukázka jednoduché RNN vrstvy v Keras
inputs = Input(shape=(None, 10))  # (batch_size, timesteps, input_features)
rnn_layer = SimpleRNN(32, return_sequences=True)(inputs)  # výstup pro každý časový krok
output_layer = Dense(1)(rnn_layer)

model = Model(inputs=inputs, outputs=output_layer)
model.summary()
```

Rekurentní vrstva lze matematicky vyjádřit jako:

```
h_t = tanh(W_x * x_t + W_h * h_{t-1} + b)
y_t = W_y * h_t + b_y
```

kde:
- h_t je skrytý stav v čase t
- x_t je vstup v čase t
- W_x, W_h, W_y jsou váhové matice
- b, b_y jsou bias vektory

## Problém mizejícího a explodujícího gradientu

Klasické RNN trpí problémem mizejícího a explodujícího gradientu, což omezuje jejich schopnost učit se dlouhodobé závislosti.

```python
import matplotlib.pyplot as plt

# Vizualizace problému mizejícího gradientu
def simulate_vanishing_gradient():
    time_steps = 100
    gradients = [1.0]
    
    # Simulace opakovaného násobení malým číslem (< 1)
    for i in range(1, time_steps):
        gradients.append(gradients[-1] * 0.7)  # 0.7 reprezentuje gradient menší než 1
    
    plt.figure(figsize=(10, 6))
    plt.plot(gradients)
    plt.title('Mizející gradient při backpropagation v RNN')
    plt.xlabel('Časový krok')
    plt.ylabel('Velikost gradientu')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

simulate_vanishing_gradient()
```

## LSTM: Long Short-Term Memory sítě

LSTM sítě byly navrženy speciálně pro řešení problému mizejícího gradientu a umožňují efektivní učení dlouhodobých závislostí.

### Architektura LSTM buňky

LSTM buňka obsahuje:
- Vstupní bránu (input gate)
- Zapomínací bránu (forget gate)
- Výstupní bránu (output gate)
- Buněčný stav (cell state)

```python
from tensorflow.keras.layers import LSTM

# Ukázka LSTM vrstvy v Keras
inputs = Input(shape=(None, 10))  # (batch_size, timesteps, features)
lstm_layer = LSTM(64, return_sequences=True)(inputs)
output_layer = Dense(1)(lstm_layer)

lstm_model = Model(inputs=inputs, outputs=output_layer)
lstm_model.summary()
```

### LSTM vs. SimpleRNN - Porovnání schopnosti učení dlouhodobých závislostí

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Generování syntetických dat - zpožděné kopírování
def generate_copy_data(n_samples=1000, seq_length=20, delay=5):
    X = np.zeros((n_samples, seq_length, 1))
    Y = np.zeros((n_samples, seq_length, 1))
    
    for i in range(n_samples):
        # Náhodný vektor hodnot mezi 0 a 1
        signal = np.random.rand(seq_length - delay, 1)
        
        # Vkládáme hodnoty jako vstup
        X[i, :seq_length-delay, 0] = signal.flatten()
        
        # Výstup je zpožděný vstup
        Y[i, delay:, 0] = signal.flatten()
    
    return X, Y

# Generování dat
X, Y = generate_copy_data(n_samples=1000, seq_length=50, delay=10)

# SimpleRNN model
rnn_model = Sequential([
    SimpleRNN(32, return_sequences=True, input_shape=(None, 1)),
    Dense(1)
])
rnn_model.compile(optimizer='adam', loss='mse')

# LSTM model
lstm_model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(None, 1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

# Trénink modelů
rnn_history = rnn_model.fit(X, Y, batch_size=32, epochs=20, validation_split=0.2, verbose=0)
lstm_history = lstm_model.fit(X, Y, batch_size=32, epochs=20, validation_split=0.2, verbose=0)

# Vizualizace průběhu trénování
plt.figure(figsize=(10, 6))
plt.plot(rnn_history.history['loss'], label='SimpleRNN Training Loss')
plt.plot(rnn_history.history['val_loss'], label='SimpleRNN Validation Loss')
plt.plot(lstm_history.history['loss'], label='LSTM Training Loss')
plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')
plt.title('SimpleRNN vs LSTM - Schopnost učení dlouhodobých závislostí')
plt.xlabel('Epocha')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()
```

## Implementace RNN a LSTM v TensorFlow/Keras

### Predikce časových řad s LSTM

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Generování syntetické časové řady (sinusová vlna + šum)
def generate_time_series(n_samples=1000):
    time = np.linspace(0, 100, n_samples)
    series = 0.5 * np.sin(0.1 * time) + 0.1 * np.sin(0.3 * time)
    series += 0.1 * np.random.randn(n_samples)  # šum
    return series

# Příprava dat ve formě pohybujícího se okna
def create_dataset(series, look_back=10):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i + look_back])
        y.append(series[i + look_back])
    return np.array(X), np.array(y)

# Generování a příprava dat
series = generate_time_series()
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series.reshape(-1, 1))

look_back = 20
X, y = create_dataset(series_scaled, look_back)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Rozdělení na trénovací a testovací sady
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Vytvoření modelu LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

# Predikce
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverzní transformace pro získání původní škály
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Vizualizace výsledků
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(series_scaled), label='Skutečná data')
plt.plot(range(look_back, look_back + len(train_predict)), train_predict, label='Predikce na trénovacích datech')
plt.plot(range(look_back + len(train_predict), look_back + len(train_predict) + len(test_predict)), 
         test_predict, label='Predikce na testovacích datech')
plt.title('LSTM Predikce časové řady')
plt.xlabel('Čas')
plt.ylabel('Hodnota')
plt.legend()
plt.show()
```

### Sentiment analýza textu s LSTM

```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Načtení IMDB datasetu
max_features = 10000  # počet slov ve slovníku
maxlen = 200  # počet slov v každé recenzi

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Padding sekvencí (doplnění nulami, aby všechny měly stejnou délku)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Vytvoření modelu pro sentiment analýzu
model = Sequential([
    Embedding(max_features, 128),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Trénování modelu
history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=10,
                    validation_data=(x_test, y_test))

# Vyhodnocení modelu
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {score[1]*100:.2f}%")

# Vizualizace průběhu trénování
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

## Implementace RNN a LSTM v PyTorch

### Jednoduchá RNN implementace v PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Definice RNN modelu v PyTorch
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        if hidden is None:
            batch_size = x.size(0)
            hidden = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        
        output, hidden = self.rnn(x, hidden)
        output = self.linear(output)
        
        return output, hidden

# Generování jednoduchých dat pro testování
def generate_sine_wave(seq_length=100, num_samples=1000):
    x = np.linspace(0, 20 * np.pi, seq_length)
    sines = []
    for _ in range(num_samples):
        phase = np.random.rand() * 2 * np.pi
        sine = np.sin(x + phase)
        sines.append(sine)
    return np.array(sines)

# Příprava dat
seq_length = 100
num_samples = 1000
look_back = 10

sines = generate_sine_wave(seq_length, num_samples)
X = np.zeros((num_samples, seq_length - look_back, look_back, 1))
Y = np.zeros((num_samples, seq_length - look_back, 1))

for i in range(num_samples):
    for j in range(seq_length - look_back):
        X[i, j, :, 0] = sines[i, j:j+look_back]
        Y[i, j, 0] = sines[i, j+look_back]

# Konverze do PyTorch tenzorů
X_tensor = torch.FloatTensor(X)
Y_tensor = torch.FloatTensor(Y)

# Rozdělení na trénovací a testovací sady
train_size = int(0.8 * num_samples)
X_train = X_tensor[:train_size]
Y_train = Y_tensor[:train_size]
X_test = X_tensor[train_size:]
Y_test = Y_tensor[train_size:]

# Inicializace modelu, loss funkce a optimalizátoru
model = SimpleRNN(input_size=1, hidden_size=32, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trénování modelu
num_epochs = 20
batch_size = 32
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    # Trénování po dávkách
    for i in range(0, train_size, batch_size):
        if i + batch_size > train_size:
            break
            
        x_batch = X_train[i:i+batch_size]
        y_batch = Y_train[i:i+batch_size]
        
        # Flatten vzorky a timesteps
        batch_size, timesteps, lookback, features = x_batch.size()
        x_batch = x_batch.reshape(batch_size * timesteps, lookback, features)
        y_batch = y_batch.reshape(batch_size * timesteps, -1)
        
        # Forward pass
        optimizer.zero_grad()
        y_pred, _ = model(x_batch)
        loss = criterion(y_pred, y_batch)
        
        # Backward pass a optimalizace
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
        
    epoch_loss = running_loss / num_batches
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}')

# Vizualizace průběhu trénování
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.title('RNN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
```

### LSTM model pro generování textu v PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import string
import random

# Definice LSTM modelu pro generování textu
class TextGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(TextGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden=None):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size)
            
        output, hidden = self.lstm(encoded, hidden)
        output = self.decoder(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                torch.zeros(self.n_layers, batch_size, self.hidden_size))

# Příprava jednoduchého datasetu pro demonstraci
text = """
Rekurentní neuronové sítě jsou mocným nástrojem pro zpracování sekvenčních dat.
LSTM sítě řeší problém mizejícího gradientu pomocí bran, které umožňují efektivnější učení dlouhodobých závislostí.
"""

# Vytvoření slovníku znaků
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# Příprava dat pro trénování
seq_length = 20
X = []
y = []

for i in range(len(text) - seq_length):
    X.append([char_to_idx[ch] for ch in text[i:i+seq_length]])
    y.append(char_to_idx[text[i+seq_length]])

# Konverze do PyTorch tenzorů
X_tensor = torch.LongTensor(X)
y_tensor = torch.LongTensor(y)

# Trénování modelu
hidden_size = 128
model = TextGenerator(vocab_size, hidden_size, vocab_size, n_layers=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    batches = 0
    
    for i in range(0, len(X), batch_size):
        if i + batch_size > len(X):
            break
        
        x_batch = X_tensor[i:i+batch_size]
        y_batch = y_tensor[i:i+batch_size]
        
        # Forward pass
        optimizer.zero_grad()
        output, hidden = model(x_batch)
        
        # Reshape output for loss calculation
        loss = criterion(output[:, -1, :], y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batches += 1
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/batches:.4f}')

# Generování textu pomocí natrénovaného modelu
def generate_text(model, start_string, length=100):
    model.eval()
    input_seq = [char_to_idx[ch] for ch in start_string]
    
    with torch.no_grad():
        hidden = None
        generated_text = start_string
        
        for _ in range(length):
            # Připravíme vstup
            if len(input_seq) > seq_length:
                input_seq = input_seq[-seq_length:]
                
            input_tensor = torch.LongTensor([input_seq])
            
            # Predikce
            output, hidden = model(input_tensor, hidden)
            output = output[:, -1, :]  # Zajímá nás pouze poslední výstup
            
            # Vzorkování dalšího znaku
            probs = torch.softmax(output, dim=1)
            next_char_idx = torch.multinomial(probs, 1).item()
            
            # Přidání znaku do výsledného textu
            generated_text += idx_to_char[next_char_idx]
            input_seq.append(next_char_idx)
        
        return generated_text

# Ukázka generování textu
start_string = "Rekurentní"
print(generate_text(model, start_string, length=200))
```

## Praktické aplikace a příklady

### Aplikační oblasti rekurentních sítí

1. **Zpracování přirozeného jazyka (NLP)**
   - Strojový překlad
   - Sentiment analýza
   - Sumarizace textu
   - Chatboty a konverzační agenti

2. **Zpracování časových řad**
   - Predikce finančních trhů
   - Předpověď poptávky
   - Analýza průmyslových senzorů

3. **Zpracování zvuku a řeči**
   - Rozpoznávání řeči
   - Generování hudby
   - Identifikace mluvčího

4. **Bioinformatika**
   - Analýza DNA sekvencí
   - Predikce proteinových struktur

### Multivariantní časová řada - příklad predikce

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Generování multivariantní časové řady
def generate_multivariate_data(n_samples=1000, n_features=3):
    time = np.linspace(0, 100, n_samples)
    data = np.zeros((n_samples, n_features))
    
    # Feature 1: Sinusová vlna
    data[:, 0] = 0.5 * np.sin(0.1 * time)
    
    # Feature 2: Kosinusová vlna s jinou frekvencí
    data[:, 1] = 0.3 * np.cos(0.3 * time)
    
    # Feature 3: Kombinace + trend
    data[:, 2] = 0.2 * np.sin(0.2 * time) + 0.002 * time
    
    # Přidání šumu ke všem příznakům
    data += 0.05 * np.random.randn(n_samples, n_features)
    
    return pd.DataFrame(data, columns=[f'feature_{i+1}' for i in range(n_features)])

# Vytvoření datasetu s pohybujícím se oknem
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].values
        y = data.iloc[i + seq_length].values
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Generování dat
df = generate_multivariate_data(n_samples=1000, n_features=3)

# Normalizace dat
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Vytvoření sekvencí
seq_length = 20
X, y = create_sequences(df_scaled, seq_length)

# Rozdělení na trénovací a testovací data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Vytvoření a trénování modelu
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(25, activation='relu'),
    Dense(y_train.shape[1])
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                    validation_data=(X_test, y_test), verbose=0)

# Predikce a inverzní transformace
y_pred = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# Vizualizace výsledků pro první příznak
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv[:100, 0], label='Skutečné hodnoty')
plt.plot(y_pred_inv[:100, 0], label='Predikované hodnoty')
plt.title('Predikce multivariantní časové řady pomocí LSTM')
plt.xlabel('Časový krok')
plt.ylabel('Hodnota (feature_1)')
plt.legend()
plt.show()

# Vizualizace trénovací a validační loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Trénovací loss')
plt.plot(history.history['val_loss'], label='Validační loss')
plt.title('Model loss')
plt.xlabel('Epocha')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

## Shrnutí a nejlepší postupy

### Výhody a limity RNN a LSTM
- **Výhody**
  - Efektivní zpracování sekvenčních dat
  - Schopnost modelovat časové závislosti
  - Flexibilní délka vstupních sekvencí
  - LSTM umí zachytit dlouhodobé závislosti

- **Limity**
  - Vyšší výpočetní náročnost
  - Složitější proces trénování
  - Tendence k přetrénování na malých datasetech
  - Sekvenční povaha znemožňuje paralelizaci

### Nejlepší postupy pro práci s RNN a LSTM
- Používejte vrstvy dropout pro prevenci přetrénování
- Využívejte gradient clipping pro stabilní trénink
- Zvažte obousměrné RNN pro úlohy, kde je důležitý kontext v obou směrech
- Experimentujte s počtem vrstev a velikostí skrytého stavu
- Pro velmi dlouhé sekvence zvažte alternativy jako Transformery

### Kdy použít RNN nebo LSTM
- **RNN**: Pro krátké sekvence s jednoduchými závislostmi
- **LSTM/GRU**: Pro delší sekvence s komplexními dlouhodobými závislostmi
- **Bidirectional LSTM**: Když je důležitý kontext v obou směrech (např. NLP)
- **Transformers**: Pro velmi dlouhé sekvence nebo pokud je paralelizace klíčová

---

## Shrnutí

Rekurentní neuronové sítě (RNN) a jejich varianta LSTM (Long Short-Term Memory) představují klíčové architektury pro zpracování sekvenčních dat v oblasti deep learningu. Standardní RNN dokáží modelovat časové závislosti v datech, ale trpí problémem mizejícího a explodujícího gradientu, což omezuje jejich schopnost učit se dlouhodobé závislosti.

LSTM sítě byly navrženy jako řešení těchto problémů pomocí sofistikovaného systému bran (input, forget, output gate), které umožňují síti selektivně uchovávat nebo zapomínat informace. Tato architektura se prokázala jako mimořádně efektivní při úlohách jako predikce časových řad, zpracování přirozeného jazyka, rozpoznávání řeči a mnoha dalších aplikacích.

Moderní implementace RNN a LSTM využívají frameworky jako TensorFlow/Keras nebo PyTorch, které nabízejí vysokoúrovňové API pro snadnou implementaci a trénink těchto sítí. Pro efektivní trénování je klíčová správná příprava dat, volba hyperparametrů a použití regularizačních technik.

I když RNN a LSTM byly dlouho dominantními architekturami pro sekvenční data, v posledních letech je v mnoha úlohách předčily nové architektury založené na mechanismu pozornosti (attention), zejména Transformery. Přesto zůstávají RNN a LSTM důležitou součástí toolkitu pro práci se sekvenčními daty a časovými řadami v mnoha praktických aplikacích.