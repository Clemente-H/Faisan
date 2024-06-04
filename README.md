
# Faisan: Mejora de Capacidades del Modelo Falcon-7B para el Español

Faisan es una extensión del modelo de lenguaje Falcon-7B, originalmente parte de la familia Falcon desarrollada por el Technology Innovation Institute de Abu Dhabi. Esta versión se especializa en mejorar la comprensión y generación de texto en español mediante un proceso de pre-entrenamiento secundario y ajuste fino con datos en español.

## Innovaciones
- **Preentrenamiento Secundario y Fine-Tuning:** Utilizamos un conjunto diverso de corpus no anotados en español y otros datos específicos para ajustar el modelo, omitiendo técnicas como token augmentation.
- **Evaluación con MT-Bench Español:** Proporcionamos una adaptación del MT-Bench al español para evaluar el rendimiento de Faisan en comparación con otros modelos de gran tamaño.

## Descargas
- [Faisan-7B](https://huggingface.co/ClementeH/faisan-7b)
- [Faisan-7B Instruct](https://huggingface.co/ClementeH/Faisan-7b-Instruct-v3)

## Conjuntos de Datos
- **Spanish Unannotated Corpora**: [Repositorio GitHub](https://github.com/josecannete/spanish-corpora)
- **Standford Alpaca Dataset**: [Hugging Face Datasets](https://huggingface.co/datasets/tatsu-lab/alpaca)
- **Dolly-15K**: [Hugging Face Datasets](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- **Open Orca**: [Hugging Face Datasets](https://huggingface.co/datasets/Open-Orca/OpenOrca)

## Resultados de Evaluación
Los resultados muestran una mejora en la comprensión y generación de texto en español, con un incremento de hasta 2.6 puntos en MT-Bench en algunas tareas.

## Conclusiones
El modelo Faisan-7B muestra mejoras significativas en la comprensión y generación de texto en español. Aunque no supera en todas las tareas al modelo Falcon-7B original en inglés, demuestra la capacidad de funcionar eficazmente en escenarios específicos como el roleplay en español.
