Actúa como un Ingeniero DevOps Experto y Arquitecto de Software en Python.

Tengo un chatbot desarrollado en Python y necesito prepararlo para su despliegue en producción utilizando Docker. 

Por favor, genera un PROMPT DETALLADO Y EXTENSO que yo pueda utilizar (o que puedas ejecutar tú mismo a continuación) para implementar esta contenedorización. El prompt que vas a construir debe exigir el cumplimiento estricto de los siguientes requerimientos técnicos:

### REQUERIMIENTOS TÉCNICOS A INTEGRAR EN EL PROMPT:
1. **Dockerfile Optimizado:** Creación de un `Dockerfile` utilizando una imagen base de Python ligera (como `python:3.11-slim` o similar) para minimizar el tamaño y los tiempos de construcción.
2. **Gestión de Dependencias:** Estrategia eficiente para copiar y ejecutar `requirements.txt` aprovechando el caché de capas de Docker.
3. **Seguridad y Permisos:** Configuración del contenedor para que se ejecute con un usuario "non-root" por motivos de seguridad.
4. **Variables de Entorno:** Gestión segura de configuraciones sensibles (API keys, tokens, puertos) a través de `.env` y variables de entorno, sin exponerlas en el código fuente.
5. **Orquestación Básica:** Inclusión de un archivo `docker-compose.yml` para facilitar levantar y bajar el entorno con un solo comando, dejándolo preparado por si el chatbot necesita conectarse a una base de datos en el futuro.

### RESTRICCIÓN DE EJECUCIÓN (CRÍTICO):
El prompt debe obligar al modelo a trabajar en dos fases estrictas:
- **Fase 1: Planificación del Despliegue:** Antes de escribir una sola línea de configuración, el modelo debe presentar un plan detallado. Debe explicar qué imagen base utilizará, cómo estructurará el directorio de trabajo en el contenedor y cómo manejará las variables de entorno. No mostrará código ni archivos `.yml` hasta que el usuario apruebe el plan.
- **Fase 2: Implementación:** Una vez aprobado el plan, procederá a generar el `Dockerfile`, el `.dockerignore`, el `docker-compose.yml` y los comandos exactos de terminal (`docker build`, `docker-compose up -d`) para poner en marcha el chatbot.