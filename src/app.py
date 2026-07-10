import os
import pandas as pd
# Importamos las herramientas de LangChain para embeddings y base de datos vectorial
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def iniciar_base_vectorial(ruta_csv):
    """
    Cumple con la Etapa 2 (Extracción) y Etapa 3 (Indexación Vectorial).
    Lee el CSV, genera los embeddings y los guarda en una base de datos indexada.
    """
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta_csv}")
    
    # --- ETAPA 2: EXTRACCIÓN Y PROCESAMIENTO ---
    df = pd.read_csv(ruta_csv)
    df = df.dropna(subset=['pregunta', 'respuesta'])
    
    documentos_langchain = []
    
    for _, fila in df.iterrows():
        # Combinamos pregunta y respuesta para el contenido que buscará la IA
        contenido_texto = f"Pregunta: {fila['pregunta']}\nRespuesta: {fila['respuesta']}"
        
        # Guardamos los metadatos de forma explícita (Categoría y Responsable)
        metadatos = {
            "id": int(fila['id']),
            "categoria": fila['categoria'],
            "responsable": fila['responsable']
        }
        
        # Creamos el objeto Document que LangChain entiende perfectamente
        doc = Document(page_content=contenido_texto, metadata=metadatos)
        documentos_langchain.append(doc)
    
    # --- ETAPA 3: EMBEDDINGS E INDEXACIÓN ---
    print("⏳ Generando embeddings numéricos e indexando en la base de datos vectorial...")
    
    # Usamos un modelo gratuito, ligero y profesional de HuggingFace para generar los vectores
    modelo_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Creamos la base de datos vectorial Chroma en memoria (rápida y sin configurar servidores externos)
    base_vectorial = Chroma.from_documents(
        documents=documentos_langchain,
        embedding=modelo_embeddings
    )
    
    print("✅ ¡Indexación completada con éxito! Los metadatos y vectores están listos.")
    return base_vectorial

# Sección de prueba para que veas la "magia" de la similitud semántica
if __name__ == "__main__":
    ruta = "data/politica_interna.csv"
    try:
        # Inicializamos la base de datos con nuestros documentos
        db_vectorial = iniciar_base_vectorial(ruta)
        
        # PROBAMOS LA BÚSQUEDA SEMÁNTICA
        # Nota que no usamos las palabras exactas del CSV ("plazo", "reembolso")
        pregunta_empleado = "¿Cuánto tiempo tengo para pedir que me devuelvan mi dinero?"
        
        print(f"\n🔍 Buscando en la base vectorial para la pregunta: '{pregunta_empleado}'")
        
        # Le pedimos a la base de datos que nos traiga el documento más cercano (k=1)
        resultados = db_vectorial.similarity_search(pregunta_empleado, k=1)
        
        if resultados:
            documento_encontrado = resultados[0]
            print("\n📌 Fragmento más relevante encontrado:")
            print(documento_encontrado.page_content)
            print(f"DMetadatos asociados -> Área: {documento_encontrado.metadata['categoria']} | Responsable: {documento_encontrado.metadata['responsable']}")
            
    except Exception as e:
        print(f"❌ Error en el proceso: {e}")
