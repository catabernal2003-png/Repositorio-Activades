# Repositorio-Activades
En este repositorio se desarrollaran todos los ejercicios propuestos en clase

##Aprendizaje por Refuerzo — Implementación Q-Learning con Flask

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
