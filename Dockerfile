# Usa una imagen de Python oficial como base
FROM python:3.10
# Establece el directorio de trabajo en /app
WORKDIR /proyecto_docker
# Copia el archivo requirements.txt al directorio de trabajo
COPY requirements.txt .
# Instala las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt
# Copia el contenido de la carpeta actual en el directorio de trabajo
COPY . .
# Exponer el puerto en el que Streamlit se ejecutará (por defecto 8501)
EXPOSE 8501
# Comando para ejecutar la aplicación Streamlit
CMD ["streamlit", "run", "app.py"]
