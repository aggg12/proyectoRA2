# ALE-LIBRARY (Posiblemente EvoNet) 🧬🧠
**High-Performance Evolutionary Neural Network Library in C++17**

**ALE-LIBRARY** (internally powered by `evonet`) es una biblioteca ligera, modular y de alto rendimiento diseñada para crear, entrenar y evolucionar redes neuronales artificiales utilizando algoritmos genéticos y neuroevolución. 

A diferencia de las bibliotecas tradicionales como TensorFlow o PyTorch que usan Backpropagation (Descenso de gradiente), ALE-LIBRARY utiliza **Computación Evolutiva**, lo que permite entrenar redes en escenarios donde no hay gradientes diferenciables o se busca optimizar la topología de la red.

---

## 🚀 Características Principales

*   **Cero Dependencias:** Escrito completamente en C++17 estándar (STL). No requiere instalar nada extra.
*   **Doble Motor de IA:**
    1.  **Standard GA-DL:** Redes de Topología Fija (Deep Learning clásico) entrenadas con Algoritmos Genéticos. Ideal cuando ya conoces la arquitectura que necesitas.
    2.  **NEAT (NeuroEvolution of Augmenting Topologies):** Redes que empiezan desde cero y hacen crecer sus neuronas y conexiones evolutivamente. Ideal cuando quieres la solución más eficiente y pequeña posible.
*   **Curriculum Learning (Aprendizaje Adaptativo):** Sistema inteligente que aumenta la dificultad (ruido) automáticamente solo cuando la IA domina el nivel actual.
*   **Smart Saving:** Sistema de guardado que prioriza la robustez. Nunca perderás tu mejor modelo; guarda automáticamente al romper récords de dificultad.
*   **Entrenamiento Infinito:** Ciclos de entrenamiento perpetuos (hasta `Ctrl+C`) que permiten refinar modelos durante horas o días.

---

## 📦 Instalación

Simplemente clona el repositorio. No necesitas instalar librerías de Python ni drivers de CUDA. Solo necesitas un compilador de C++ (GCC o Clang) y `make`.

```bash
git clone https://github.com/tu-usuario/ALE-LIBRERY.git
cd ALE-LIBRERY
make
```

Esto compilará automáticamente todos los ejemplos y herramientas de entrenamiento.

---

## 🛠️ Guía de Uso Rápido

La librería viene con un **Makefile** robusto que gestiona todo el ciclo de vida del entrenamiento.

### 1. Entrenar Redes Profundas (Standard DL)
Usa este modo si quieres definir tú mismo las capas (ej. "quiero 2 capas ocultas de 60 y 30 neuronas").

```bash
make train_dl
# Ejecuta el entrenamiento infinito. Detenlo con Ctrl+C cuando estés satisfecho.
```
*   **Archivo principal:** `examples/02_deep_learning/main_digits.cpp`
*   **Guardado:** Crea un archivo `digits_model.txt` (o similar) con el mejor cerebro.

### 2. Entrenar con NEAT (Evolución de Topología)
Usa este modo si quieres que la IA descubra sola cuántas neuronas necesita. Empezará con una red vacía y crecerá.

```bash
make train_neat
```
*   **Archivo principal:** `examples/03_neat/main_digits.cpp`
*   **Guardado:** Crea un archivo `neat_digits_model.txt`.

### 3. Probar la Robustez
Una vez entrenados, puedes someter a tus IAs a pruebas de estrés con niveles de ruido que nunca han visto.

```bash
make test_dl    # Prueba la red Standard
make test_neat  # Prueba la red NEAT
```

---

## ⚙️ Configuración y Personalización

Todo el comportamiento de la librería es configurable desde el código C++.

### Configuración Standard (`GeneticAlgorithm`)
En `examples/02_deep_learning/main_digits.cpp`:

```cpp
// Definir arquitectura: [Inputs, Oculta1, Oculta2, ..., Outputs]
GeneticAlgorithm ga(200, {35, 60, 30, 10}); 
// 200 = Tamaño de población (cuantos "pupilos" compiten a la vez)
```

### Configuración NEAT (`NeatConfig`)
En `examples/03_neat/main_digits.cpp`. NEAT tiene muchos parámetros para ajustar la evolución:

```cpp
NeatConfig cfg;
cfg.populationSize = 300;       // Cantidad de especies/organismos
cfg.addConnectionRate = 0.40;   // Probabilidad de crear una nueva sinapsis
cfg.addNodeRate = 0.10;         // Probabilidad de crear una nueva neurona (complejidad)
cfg.weightMutationRate = 0.8;   // Frecuencia de cambio de pesos
cfg.survivalRate = 0.20;        // Solo el top 20% sobrevive para reproducirse
```

### Control de Dificultad (Curriculum)
Puedes ajustar qué tan difícil se vuelve el entrenamiento editando el límite de ruido en los archivos `main`.

```cpp
// Ejemplo en main_digits.cpp
if (maxHits >= 9 && noiseLevel < 0.20) { // <--- TOPE DE RUIDO (20%)
    // Lógica de subida de nivel...
}
```

---

## 🧪 Documentación Técnica de Clases

Para aprovechar al máximo ALE-LIBRARY, tienes control total sobre los dos motores.

### 1. `GeneticAlgorithm` (Deep Learning Standard)
Ubicación: `include/evonet/standard/GeneticAlgorithm.hpp`

Esta clase gestiona una población fija de redes neuronales con topología idéntica (capas y neuronas predefinidas).

#### Constructor
```cpp
GeneticAlgorithm(
    size_t popSize,                  // Número de agentes (ej. 200)
    const std::vector<size_t>& topo, // Arquitectura (ej. {35, 60, 10})
    double mutRate = 0.05,           // Prob. de mutación de un peso (5%)
    double crossRate = 0.5,          // Prob. de cruce (50%)
    size_t elitism = 2               // Nº de campeones intocables por generación
);
```

#### Métodos Clave
*   `evolve()`: Ejecuta una generación completa (Selección -> Cruce -> Mutación). Llámalo después de haber calculado el `fitness` de cada agente.
*   `getPopulation()`: Devuelve el `vector<Agent>` para que puedas iterar sobre ellos y probarlos (feed-forward).
*   `setActivationType(ActivationType type)`: Cambia la función de activación de toda la población (`SIGMOID`, `RELU`, `TANH`, `LINEAR`).

---

### 2. `NeatConfig` y `Population` (NEAT)
Ubicación: `include/evonet/neat/`

Este motor es más complejo. Usa `Population` para gestionar especies que compiten en nichos ecológicos.

#### Estructura `NeatConfig`
Es el panel de control de la evolución. Modifica sus campos antes de crear la `Population`.

**Probabilidades de Mutación Estructural:**
*   `addConnectionRate` (Default 0.05): Probabilidad de unir dos neuronas desconectadas.
*   `addNodeRate` (Default 0.03): Probabilidad de partir una conexión y crear una neurona nueva en medio.
*   `weightMutationRate` (Default 0.80): Frecuencia con la que cambian los pesos existentes.

**Gestión de Especies:**
*   `compatibilityThreshold`: Cuán diferentes deben ser dos redes para ser especies distintas.
*   `targetSpeciesCount`: El sistema ajustará dinámicamente el umbral para intentar mantener este número de especies vivas.
*   `survivalRate` (Default 0.5): Porcentaje de individuos que sobreviven dentro de cada especie.

#### Métodos Clave de `Population`
*   `evolve()`: Gestiona todo el ciclo de vida: especiación, ajuste dinámico de umbrales, eliminación de especies estancadas (`dropOffAge`) y reproducción.
*   `getGenomes()`: Devuelve todos los organismos actuales.
*   `getBestGenome()`: Devuelve un puntero al mejor organismo global de la generación actual.

---

## 📂 Estructura del Proyecto

Para integrar esta librería en tu propio proyecto, solo necesitas la carpeta `include/evonet`.

```text
ALE-LIBRERY/
├── include/
│   └── evonet/
│       ├── Matrix.hpp            # Motor matemático (sin dependencias)
│       ├── DigitRecognition.hpp  # Dataset de ejemplo (Dígitos 5x7)
│       ├── standard/             # IA Clásica (Topología Fija)
│       │   ├── NeuralNetwork.hpp # La red neuronal
│       │   ├── GeneticAlgorithm.hpp # El entrenador genético
│       │   └── Agent.hpp         # Contenedor para la población
│       └── neat/                 # IA Avanzada (Evolutiva)
│           ├── Genome.hpp        # Genotipo (ADN de la red)
│           ├── Network.hpp       # Fenotipo (La red ejecutable)
│           ├── Population.hpp    # Gestor de especies y evolución
│           └── InnovationDatabase.hpp # Histórico genealógico
├── examples/
│   ├── 01_basics/                # Pruebas simples (XOR, Seno)
│   ├── 02_deep_learning/         # Reconocimiento de Patrones (Standard)
│   └── 03_neat/                  # Reconocimiento de Patrones (NEAT)
└── Makefile                      # Comandos de compilación
```

---

## 🧠 ¿Por qué es tan rápido?

ALE-LIBRARY está optimizado para **CPU**.
1.  **Redes Pequeñas ("TinyML"):** Al evolucionar topologías, las redes resultantes suelen ser 100x más pequeñas que las redes profundas tradicionales, requiriendo menos operaciones matemáticas.
2.  **Sin Overhead:** Cada ciclo de CPU se usa para calcular neuronas. No hay capas de abstracción pesadas como en Python/Keras.
3.  **Compilación -O3:** Aprovecha la vectorización y optimización de bucles del compilador C++.

---

## 📄 Licencia

Este proyecto es de código abierto. Siéntete libre de modificar, aprender y evolucionar tus propias inteligencias artificiales.
