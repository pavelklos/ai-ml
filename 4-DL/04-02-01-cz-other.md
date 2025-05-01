# Konvoluční neuronové sítě (CNN) - architektura

## Obsah
1. Úvod do konvolučních neuronových sítí
2. Základní komponenty CNN architektury
3. Populární CNN architektury
4. Implementace CNN v Pythonu
5. Vizualizace a porozumění CNN
6. Shrnutí a nejlepší postupy

## Úvod do konvolučních neuronových sítí

Konvoluční neuronové sítě (CNN) jsou specializovaným typem neuronových sítí navržených především pro zpracování dat s mřížkovou topologií, jako jsou obrázky. Jejich architektura je inspirována organizací vizuálního kortexu u savců, kde jednotlivé neurony reagují na podněty pouze v omezené oblasti zorného pole.

**Klíčové vlastnosti CNN:**
- Lokální vnímání vzorů pomocí konvolučních filtrů
- Sdílení parametrů napříč celým vstupním prostorem
- Hierarchická extrakce příznaků od jednoduchých po komplexní
- Invariance vůči posunu díky pooling operacím

## Základní komponenty CNN architektury

### Konvoluční vrstvy

Konvoluční vrstvy jsou základním stavebním blokem CNN. Provádějí operaci konvoluce mezi vstupem a filtry (kernely) pro detekci lokálních vzorů.

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

# Vytvoření 2D konvoluční vrstvy
conv_layer = Conv2D(
    filters=32,              # Počet výstupních filtrů
    kernel_size=(3, 3),      # Velikost konvolučního jádra
    strides=(1, 1),          # Krok posunutí filtru
    padding='same',          # Typ paddingu ('valid' nebo 'same')
    activation='relu',       # Aktivační funkce
    input_shape=(28, 28, 1)  # Vstupní tvar pro první vrstvu (výška, šířka, kanály)
)
```

### Aktivační funkce

Aktivační funkce zavádějí nelinearitu do sítě, což je klíčové pro učení komplexních vzorů.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Běžné aktivační funkce v CNN
x = np.linspace(-5, 5, 100)

# ReLU - nejčastěji používaná v CNN
relu = np.maximum(0, x)

# Leaky ReLU - varianta ReLU bez mrtvých neuronů
leaky_relu = np.maximum(0.1*x, x)

# Vizualizace
plt.figure(figsize=(10, 6))
plt.plot(x, relu, label='ReLU')
plt.plot(x, leaky_relu, label='Leaky ReLU')
plt.grid(True)
plt.legend()
plt.title("Aktivační funkce používané v CNN")
plt.show()
```

### Pooling vrstvy

Pooling vrstvy redukují prostorové rozměry reprezentace, snižují počet parametrů a výpočetní náročnost.

```python
from tensorflow.keras.layers import MaxPool2D, AveragePooling2D

# Max pooling - vybírá maximální hodnotu z definované oblasti
max_pool = MaxPool2D(
    pool_size=(2, 2),    # Velikost pooling okna
    strides=(2, 2),      # Krok posunutí okna
    padding='valid'      # Typ paddingu
)

# Average pooling - počítá průměr hodnot v definované oblasti
avg_pool = AveragePooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='valid'
)
```

### Plně propojené vrstvy

Plně propojené vrstvy se obvykle nacházejí na konci CNN a provádějí klasifikaci na základě získaných příznaků.

```python
from tensorflow.keras.layers import Flatten, Dense

# Zploštění vícerozměrných dat na 1D vektor
flatten = Flatten()

# Plně propojená vrstva
dense = Dense(
    units=128,           # Počet neuronů
    activation='relu',   # Aktivační funkce
)

# Výstupní vrstva pro klasifikaci (příklad pro 10 tříd)
output_layer = Dense(
    units=10,
    activation='softmax'  # Softmax pro vícetřídní klasifikaci
)
```

## Populární CNN architektury

### LeNet-5
První úspěšná CNN architektura, navržená Yannem LeCunem pro rozpoznávání ručně psaných číslic.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

def create_lenet5():
    model = Sequential([
        Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=(32, 32, 1), padding='same'),
        AveragePooling2D(pool_size=(2, 2), strides=2),
        Conv2D(16, kernel_size=(5, 5), activation='tanh'),
        AveragePooling2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(120, activation='tanh'),
        Dense(84, activation='tanh'),
        Dense(10, activation='softmax')
    ])
    return model

lenet5 = create_lenet5()
lenet5.summary()
```

### VGG16
VGG16 je známá pro svou jednoduchost a hloubku s použitím menších konvolučních filtrů (3x3).

```python
from tensorflow.keras.applications import VGG16

# Načtení předtrénovaného modelu VGG16
vgg_model = VGG16(weights='imagenet', include_top=True, 
                  input_shape=(224, 224, 3))
vgg_model.summary()
```

### ResNet (Residual Network)
ResNet představil reziduální bloky pro řešení problému mizejícího gradientu v hlubokých sítích.

```python
from tensorflow.keras.layers import Input, Add
from tensorflow.keras.models import Model

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    
    # První konvoluční vrstva
    y = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)
    
    # Druhá konvoluční vrstva
    y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    
    # Pokud se změnily dimenze, upravíme shortcut
    if stride > 1 or x.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(x)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    # Přidání reziduálního spojení
    y = Add()([y, shortcut])
    y = tf.keras.layers.Activation('relu')(y)
    
    return y

# Příklad použití reziduálního bloku
inputs = Input(shape=(224, 224, 3))
x = Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

# Přidání reziduálních bloků
x = residual_block(x, filters=64)
x = residual_block(x, filters=64)
x = residual_block(x, filters=128, stride=2)
x = residual_block(x, filters=128)

# Výstupní vrstvy
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = Dense(1000, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
```

## Implementace CNN v Pythonu

### Kompletní model CNN pomocí TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def create_cnn_model():
    model = Sequential([
        # První konvoluční blok
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Druhý konvoluční blok
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Klasifikační část
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')  # 10 výstupních tříd
    ])
    
    # Kompilace modelu
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Vytvoření modelu
cnn_model = create_cnn_model()
cnn_model.summary()
```

### Jednoduchý příklad trénování na MNIST datasetu

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# Načtení MNIST datasetu
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Předzpracování dat
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0  # Normalizace na [0,1]
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train, 10)  # One-hot kódování
y_test = to_categorical(y_test, 10)

# Vytvoření modelu
model = create_cnn_model()

# Trénování modelu
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_data=(x_test, y_test),
    verbose=1
)

# Vyhodnocení modelu
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {score[1]*100:.2f}%")

# Vizualizace průběhu trénování
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

### Implementace CNN v PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # První konvoluční vrstva
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Druhá konvoluční vrstva
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Plně propojené vrstvy
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # První blok
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        
        # Druhý blok
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        # Zploštění
        x = x.view(-1, 64 * 7 * 7)
        
        # Plně propojené vrstvy
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

# Inicializace modelu
model = SimpleCNN()
print(model)

# Ukázka tréninku s PyTorch (jednoduchý pseudokód)
'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
'''
```

## Vizualizace a porozumění CNN

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

# Předpokládejme, že máme natrénovaný model a obrázek
def visualize_feature_maps(model, image, layer_name):
    # Vytvoříme model, který vrátí výstupy z požadované vrstvy
    layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    
    # Získáme feature mapy
    feature_maps = layer_model.predict(np.expand_dims(image, axis=0))[0]
    
    # Vizualizujeme prvních 16 feature map
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < feature_maps.shape[-1]:
            ax.imshow(feature_maps[:,:,i], cmap='viridis')
            ax.set_title(f'Filter {i}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Příklad použití:
# visualize_feature_maps(cnn_model, x_test[0], 'conv2d_1')
```

## Shrnutí a nejlepší postupy

### Hlavní výhody CNN
- Automatická extrakce příznaků
- Snížení počtu parametrů díky sdílení vah
- Prostorová invariance díky pooling operacím
- Hierarchická reprezentace příznaků

### Nejlepší postupy pro návrh CNN
- Začínejte s osvědčenými architekturami
- Používejte batch normalizaci pro stabilnější trénování
- Aplikujte dropout pro prevenci přetrénování
- Správně nastavte learning rate a další hyperparametry
- Používejte data augmentaci pro rozšíření trénovacích dat
- Monitorujte validační metriky během trénování

---

## Shrnutí

Konvoluční neuronové sítě (CNN) představují specializovaný typ neuronových sítí optimalizovaný pro zpracování mřížkových dat, především obrazů. Jejich architektura je založena na biologicky inspirovaných principech a využívá hierarchickou extrakci příznaků pomocí konvolučních filtrů, nelineárních aktivací a podvzorkování.

Klíčové komponenty CNN zahrnují konvoluční vrstvy, aktivační funkce (především ReLU), pooling vrstvy a plně propojené vrstvy. Existuje mnoho osvědčených architektur jako LeNet-5, VGG16, ResNet nebo Inception, které se liší svou hloubkou, složitostí a návrhovými principy.

Moderní implementace CNN využívají frameworky jako TensorFlow/Keras nebo PyTorch, které poskytují vysokoúrovňové API pro snadné vytváření, trénování a nasazování těchto sítí. Pro efektivní trénování je důležité správné předzpracování dat, volba vhodných hyperparametrů a použití technik jako batch normalizace a dropout.

CNN našly široké uplatnění v počítačovém vidění, včetně klasifikace obrazu, detekce objektů, segmentace obrazu a mnoha dalších oblastech, kde jsou nyní považovány za standardní a velmi úspěšný přístup.