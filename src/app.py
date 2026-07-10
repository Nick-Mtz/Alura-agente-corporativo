import pandas as pd
import os

def cargar_y_procesar_documento(ruta_csv):
    """
    Cumple con la Etapa 2 del Challenge: Procesamiento y Extracción.
    Lee el CSV, limpia filas vacías y prepara los chunks con sus metadatos.
    """
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {ruta_csv}")
    
    # 1. Extracción por formato usando Pandas
    df = pd.read_csv(ruta_csv)
    
    # 2. Limpieza básica (eliminar filas que tengan campos vacíos)
    df = df.dropna(subset=['pregunta', 'respuesta'])
    
    chunks_procesados = []
    
    # 3. Chunking por estructura lógica (Fila por Fila) & 4. Atribución de Metadatos
    for _, fila in df.iterrows():
        # Construimos un texto altamente comprensible para el modelo de IA
        texto_chunk = (
            f"Categoría: {fila['categoria']}\n"
            f"Pregunta: {fila['pregunta']}\n"
            f"Respuesta: {fila['respuesta']}\n"
            f"Responsable del área: {fila['responsable']}"
        )
        
        # Guardamos el chunk junto con un diccionario de metadatos estructurados
        chunks_procesados.append({
            "id": fila['id'],
            "contenido": texto_chunk,
            "metadatos": {
                "categoria": fila['categoria'],
                "responsable": fila['responsable']
            }
        })
        
    return chunks_procesados

# Código de prueba interna para validar la extracción
if __name__ == "__main__":
    ruta = "data/politica_interna.csv"
    try:
        resultado = cargar_y_procesar_documento(ruta)
        print(f"✅ Extracción exitosa. Se generaron {len(resultado)} chunks listos para el agente.")
        print("\nEjemplo del primer chunk procesado:")
        print(resultado[0]["contenido"])
    except Exception as e:
        print(f"❌ Error en la extracción: {e}")
