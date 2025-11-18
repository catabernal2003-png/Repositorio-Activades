# Repositorio-Activades
En este repositorio se desarrollaran todos los ejercicios propuestos en clase

## Aprendizaje por Refuerzo — Implementación Q-Learning con Flask

Este proyecto implementa un agente de **Aprendizaje por Refuerzo (Reinforcement Learning)** usando el algoritmo **Q-Learning** dentro de un entorno tipo **GridWorld**, permitiendo entrenar, visualizar y simular el comportamiento del agente mediante una interfaz web construida con **Flask**.

---

#Contenido

- [Objetivo](#objetivo)
- [Conceptos de Aprendizaje por Refuerzo](#conceptos-de-aprendizaje-por-refuerzo)
- [Descripción del Entorno GridWorld](#descripción-del-entorno-gridworld)
- [Implementación del Agente](#implementación-del-agente)
- [Interfaz Flask](#interfaz-flask)
- [Resultados](#resultados)
- [Cómo ejecutar el proyecto](#cómo-ejecutar-el-proyecto)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Referencias](#referencias)

---

# Objetivo

Modelar un proceso de decisión secuencial donde un agente aprende a comportarse en un entorno mediante interacción, exploración y retroalimentación basada en recompensas.

El sistema:

1. Entrena un agente usando Q-Learning.
2. Visualiza el progreso del entrenamiento.
3. Permite simular la política aprendida.
4. Expone todo mediante una aplicación web Flask.

---

# Conceptos de Aprendizaje por Refuerzo

El Aprendizaje por Refuerzo consiste en entrenar un **agente** para que aprenda una política óptima mediante interacción con un **entorno**, observando **estados**, realizando **acciones**, y recibiendo **recompensas** que guían el aprendizaje.

### Elementos del modelo:
- **Agente:** quien toma decisiones.
- **Entorno:** el mundo donde el agente actúa.
- **Estado (s):** descripción del momento actual.
- **Acciones (a):** movimientos disponibles.
- **Recompensa (r):** retroalimentación inmediata.
- **Política (π):** estrategia del agente.
- **Función Q:** valor que combina estado + acción.

### Algoritmo usado: Q-Learning
Actualiza valores Q mediante:
Q(s,a) ← Q(s,a) + α [ r + γ max(Q(s', ·)) – Q(s,a) ]

Donde:
- α = tasa de aprendizaje  
- γ = factor de descuento  
- ϵ = tasa de exploración (ϵ-greedy)

---
# ⬜ Descripción del Entorno GridWorld

Se utiliza un GridWorld de 6x6:

- Estado inicial: **(0,0)**
- Meta: **(5,5)**
- Agujeros (estados penalizados):  
  - (1,3), (2,3), (3,3)
- Movimientos: arriba, abajo, izquierda, derecha
- Recompensas:
  - −1 por movimiento
  - −10 al caer en un hoyo
  - +50 al llegar a la meta

El objetivo del agente es **aprender la ruta óptima evitando los hoyos**.

---

# Implementación del agente

El archivo `rl_gridworld.py` contiene:

### ✔ Clase del entorno  
`GridWorldEnv` gestiona:
- Estados válidos  
- Transiciones  
- Recompensas  
- Grid visual  

### ✔ Algoritmo Q-Learning  
`train_q_learning()` entrena al agente con:
- Episodios configurables  
- α (learning rate)  
- γ (discount factor)  
- ϵ (exploration rate) con decay  
- Registro de recompensas  

### ✔ Simulación  
`simulate_episode()` ejecuta un episodio utilizando la política aprendida.

### ✔ Gráficos  
- `plot_rewards()` genera `rewards.png`  
- `plot_trajectory()` genera `traj.png`

---

# Interfaz Flask

La app incluye:

## ✔ Página de conceptos
Ruta: `/reinforcement/conceptos`  
Incluye teoría, esquema y referencias APA.

## ✔ Caso práctico
Ruta: `/reinforcement/caso`

Incluye:
- Formulario de parámetros de entrenamiento  
- Botón **Iniciar entrenamiento**  
- Botón **Simular política entrenada**  
- Gráfico de evolución de recompensas  
- Imagen con la trayectoria aprendida  

Toda la interfaz está estilizada con CSS personalizado.

---

# Resultados

Durante el entrenamiento, el agente:

- Reduce progresivamente su tasa de exploración.
- Mejora el valor de la recompensa acumulada.
- Aprende un camino hacia el objetivo evitando estados penalizados.

Los archivos generados incluyen:
- `/static/images/rewards.png` → gráfica de recompensa por episodio  
- `/static/images/traj.png` → recorrido aprendido  
- `models/q_table.pkl` → tabla Q entrenada  
