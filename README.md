# Product Research App

Esta aplicación permite analizar catálogos de productos y coordinar tareas con modelos de OpenAI.

## Variables de entorno

Configura estas variables para ajustar el comportamiento del orquestador de GPT:

- `OPENAI_API_KEY`: clave de OpenAI utilizada por defecto para todas las llamadas.
- `MAX_ITEMS` (por defecto `300`): tamaño máximo de lote al enviar productos al modelo en tareas de consulta y tendencias. Para imputación y desire se limita automáticamente a bloques de 100 elementos.
- `GPT_TIMEOUT` (por defecto `60` segundos): tiempo máximo de espera para las llamadas a la API de OpenAI.
- `GPT_MODEL_A` a `GPT_MODEL_E`: permiten sobreescribir los modelos por defecto utilizados en las tareas `consulta`, `pesos`, `tendencias`, `imputacion` y `desire` respectivamente.

Coloca estas variables en tu entorno o en el archivo de configuración según tus necesidades.
