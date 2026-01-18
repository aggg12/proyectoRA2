# Manual de Referencia Técnica: ALE-LIBRARY (EvoNet) 📘

**ALE-LIBRARY** es una suite de inteligencia artificial evolutiva de alto rendimiento escrita en C++17. Este documento detalla la arquitectura, configuración y uso de sus dos motores principales: **Standard GA-DL** (Deep Learning Evolutivo) y **NEAT** (Evolución de Topologías).

---

## 🏗️ 1. Motor Standard DL (GA-DL)

Este motor combina la arquitectura clásica de **Redes Neuronales Profundas (DNN/MLP)** con el entrenamiento mediante **Algoritmos Genéticos**.

### 1.1 Características Principales
*   **Topología Fija:** La estructura de la red (número de capas y neuronas) se define al inicio y no cambia. Solo evolucionan los pesos.
*   **Matricial:** Utiliza operaciones de álgebra lineal (`Matrix.hpp`) optimizadas para CPU.
*   **Ideal para:** Problemas donde ya conoces la complejidad necesaria (ej. "necesito 2 capas ocultas") o quieres replicar arquitecturas de Deep Learning tradicionales sin usar Backpropagation.

### 1.2 Referencia de API (`GeneticAlgorithm`)

La clase principal es `evonet::GeneticAlgorithm`.

#### Constructor Completo
```cpp
GeneticAlgorithm(
    size_t popSize,                  // Cantidad de agentes
    const std::vector<size_t>& topo, // Arquitectura por capas
    double mutRate = 0.05,           // Tasa de mutación (default 5%)
    double crossRate = 0.5,          // Tasa de cruce (default 50%)
    size_t elitism = 2               // Campeones intocables
);
```

| Parámetro | Default | Impacto en el Entrenamiento |
| :--- | :--- | :--- |
| `popSize` | **Req** | **Alto (e.g. 500+):** Mejor exploración, evita mínimos locales, más lento. **Bajo:** Rápido pero inestable. |
| `topo` | **Req** | Define la complejidad. `{35, 60, 30, 10}` crea una red con 35 inputs, capas ocultas de 60 y 30, y 10 outputs. |
| `mutRate` | `0.05` | Controla la "creatividad". Si es muy alto (>0.1), se vuelve aleatorio. Si es muy bajo, no aprende. |
| `elitism` | `2` | Asegura que la mejor red *nunca* se pierda. Garantiza que el fitness siempre sea monótono creciente. |

#### Métodos Útiles
*   `setActivationType(ActivationType type)`: Cambia la función de activación de toda la población (`SIGMOID`, `RELU`, `TANH`, `LINEAR`).
*   `evolve()`: Ejecuta una ronda de selección, cruce y mutación.
*   `getPopulation()`: Devuelve el vector de agentes para su evaluación.

### 1.3 Ejemplo de Implementación

```cpp
#include "evonet/standard/GeneticAlgorithm.hpp"

int main() {
    // 1. Configurar: 2 entradas -> capa oculta de 5 -> 1 salida
    std::vector<size_t> arquitectura = {2, 5, 1};
    
    // 2. Inicializar sistema con 100 agentes
    evonet::GeneticAlgorithm ga(100, arquitectura);
    
    // Opcional: Usar función de activación moderna
    ga.setActivationType(evonet::NeuralNetwork::ActivationType::RELU);

    while (true) {
        // 3. Obtener la población actual
        auto& poblacion = ga.getPopulation();

        // 4. Evaluar a cada agente (El "Examen")
        for (auto& agente : poblacion) {
            std::vector<double> inputs = {1.0, 0.0}; 
            auto output = agente.brain.feedForward(inputs);
            
            // Calcular error y asignar fitness
            double error = abs(1.0 - output[0]); // Esperamos 1.0
            agente.fitness = 10.0 - error;       // Más fitness es mejor
        }

        // 5. Evolucionar (Selección -> Cruce -> Mutación)
        ga.evolve();
    }
}
```

---

## 🧬 2. Motor NEAT (NeuroEvolution of Augmenting Topologies)

Este motor implementa el famoso algoritmo de Kenneth Stanley. **La red empieza vacía y crece sola.**

### 2.1 Características Principales
*   **Topología Dinámica:** La IA decide cuántas neuronas y conexiones necesita. Empieza simple y se vuelve compleja solo si es necesario.
*   **Especiación (Nichos):** Las redes se agrupan en especies genéticamente similares. Esto protege a las nuevas mutaciones estructurales (innovaciones) de competir injustamente con redes maduras.
*   **Orientado a Grafos:** No usa capas fijas. Las conexiones pueden ir de cualquier neurona a cualquier otra (incluso hacia atrás -> recurrencia).

### 2.2 Configuración (`NeatConfig`)

NEAT es muy flexible. Casi todo se controla desde el struct `evonet::neat::NeatConfig`.

#### Tabla de Configuración Avanzada

| Categoría | Campo | Default | Explicación |
| :--- | :--- | :--- | :--- |
| **Arquitectura** | `addConnectionRate` | `0.05` | Probabilidad de añadir un cable nuevo. Aumentar para encontrar relaciones más rápido. |
| | `addNodeRate` | `0.03` | **Crítico:** Probabilidad de dividir un cable y crear una neurona. No subir mucho o la red crecerá descontrolada. |
| **Aprendizaje** | `weightMutationRate` | `0.80` | Probabilidad de ajustar pesos existentes. Debe ser alto (80%+) para permitir aprendizaje fino. |
| **Ecosistema** | `targetSpeciesCount` | `20` | El sistema ajustará la tolerancia social para intentar mantener esta cantidad de especies distintas. |
| | `survivalRate` | `0.5` | Porcentaje de la especie que sobrevive. `0.2` es más meritocrático ("Solo lo mejor de lo mejor"). |
| | `dropOffAge` | `20` | Generaciones que se permite vivir a una especie estancada antes de extinguirla. |

### 2.3 Ejemplo de Implementación

```cpp
#include "evonet/neat/Population.hpp"
#include "evonet/neat/NeatConfig.hpp"

int main() {
    // 1. Configurar
    evonet::neat::NeatConfig cfg;
    cfg.inputCount = 2;   // Entradas
    cfg.outputCount = 1;  // Salidas
    cfg.populationSize = 150; 
    
    // Personalización: Evolución agresiva de estructura
    cfg.addConnectionRate = 0.10; // Más conexiones
    cfg.targetSpeciesCount = 10;  // Menos especies

    // 2. Crear Ecosistema
    evonet::neat::Population pop(cfg);

    while (true) {
        // En NEAT, obtenemos los genomas (planos)
        // pero necesitamos convertirlos a Redes (Networks) para ejecutarlos
        auto& genomes = pop.getGenomes();

        for (auto& genome : genomes) {
            // "Expresar" el genotipo en un fenotipo ejecutable
            evonet::neat::Network brain;
            brain.buildFromGenome(genome);
            
            std::vector<double> inputs = {1.0, 0.0};
            auto output = brain.activate(inputs);
            
            // Asignar fitness al genoma
            genome.fitness = 1.0 - abs(1.0 - output[0]);
        }

        // 3. El motor NEAT se encarga de todo (especiación, extinción, etc.)
        pop.evolve();
    }
}
```

---

## 🛠️ 3. Módulo Core (Internal)

Estas clases son utilizadas internamente, pero útiles si quieres extender la librería.

### 3.1 `Matrix` (`include/evonet/Matrix.hpp`)
Librería de álgebra lineal ligera.
*   `Matrix(rows, cols, true)`: Crea matriz con inicialización Xavier (pesos inteligentes).
*   `multiply(Matrix)`: Multiplicación $O(n^3)$ optimizada.
*   `flatten()`: Convierte la matriz a vector para manipulación genética.

### 3.2 Formatos de Guardado
ALE-LIBRARY guarda los modelos en texto plano (`.txt`) para máxima compatibilidad.

**Standard:** Guarda dimensiones seguidas de la lista plana de pesos.
**NEAT:** Guarda una lista de `NODES` (con sus tipos y bias) y una lista de `LINKS` (conexiones con sus pesos e innovaciones).

---

## 🚀 4. Estrategias Avanzadas Integradas

Los ejemplos incluidos (`main_digits.cpp`) implementan lógicas de nivel industrial que deberías copiar en tus proyectos:

### Curriculum Learning (Dificultad Progresiva)
No lances a tu IA al problema más difícil desde el día 1.
*   La librería permite definir un `noiseLevel` que controla la dificultad.
*   Solo incrementa este nivel si el `maxHits` (aciertos) es alto durante **N generaciones consecutivas**.
*   Esto crea una "rampa de aprendizaje" suave.

### Robustez Anti-Memorización
*   El código regenera el ruido aleatoriamente en **cada frame/generación**.
*   Esto obliga a la IA a generalizar principios, ya que memorizar la entrada exacta de la generación anterior no le sirve para la actual.
