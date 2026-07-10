# 🤖 Agente IA - Base de Conocimiento Corporativa Centralizada

## 📋 Descripción General
Este proyecto consiste en un **Agente de Inteligencia Artificial Corporativo** diseñado para actuar como un asistente de conocimiento centralizado y siempre disponible para todos los colaboradores de la empresa. Utilizando una arquitectura **RAG (Retrieval-Augmented Generation)**, el agente es capaz de leer, procesar e interpretar la documentación interna de la organización (en formato CSV) para responder de manera precisa, contextualizada y en lenguaje natural a las preguntas e inquietudes de los empleados sobre políticas, beneficios y procedimientos internos.

El proyecto fue desarrollado como solución al **Challenge Alura Agentes** y se encuentra completamente desplegado en la nube utilizando la infraestructura de Oracle.

---

## 🏗️ Arquitectura de la Solución
El flujo de funcionamiento del agente sigue los principios de un sistema RAG optimizado para respuestas rápidas:

1. **Ingesta de Datos:** Un script en Python lee el archivo fuente local (`data/politica_interna.csv`) que contiene las políticas y preguntas frecuentes de la empresa.
2. **Procesamiento y Contexto:** Al recibir una consulta del colaborador, el sistema localiza la información relevante dentro del documento para construir un prompt enriquecido.
3. **Generación de Respuesta (LLM):** El prompt con el contexto exacto se envía al modelo de lenguaje a través de su API para generar una respuesta fluida, verídica y libre de alucinaciones.
4. **Interfaz de Usuario:** Una interfaz web intuitiva construida con **Streamlit** permite a los empleados interactuar con el agente de forma conversacional.

---

## 🛠️ Tecnologías y Herramientas Utilizadas
* **Lenguaje:** Python 3.10+
* **Interfaz de Usuario:** Streamlit (Framework web rápido para aplicaciones de datos)
* **Procesamiento de Datos:** Pandas / Python CSV module
* **Modelos de IA:** [Especificar aquí el LLM que uses, ej. OpenAI GPT-3.5/4o, Anthropic Claude, u OCI GenAI]
* **Control de Versiones:** Git & GitHub
* **Despliegue e Infraestructura:** Oracle Cloud Infrastructure (OCI) - Compute Instance

---

## 🚀 Instrucciones para Ejecutar el Proyecto Localmente

### Prerrequisitos
* Tener instalado Python (versión 3.10 o superior).
* Contar con una API Key del proveedor de LLM seleccionado.

### Paso a Paso

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
   cd TU_REPOSITORIO
