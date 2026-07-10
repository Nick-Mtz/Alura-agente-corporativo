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

# Se añade codigo para recuperación de texto
def recuperar_contexto_relevante(base_vectorial, pregunta_usuario, categoria_filtro=None):
    """
    Cumple con la Etapa 4 del Challenge: Capa de Recuperación (RAG).
    1. Transforma la pregunta en embedding.
    2. Realiza la búsqueda semántica.
    3. Aplica filtrado por metadatos si se solicita.
    4. Ensambla el contexto final para el LLM.
    """
    # Configurar filtros por metadatos (Punto 3 de la tarjeta de Trello)
    # ChromaDB permite filtrar usando un diccionario simple
    filtros = None
    if categoria_filtro and categoria_filtro != "Todas":
        filtros = {"categoria": categoria_filtro}
    
    # Realizar la búsqueda semántica trayendo los 3 fragmentos más relevantes (k=3)
    # Pasamos el parámetro 'filter' para el filtrado por metadatos nativo
    documentos_recuperados = base_vectorial.similarity_search(
        pregunta_usuario, 
        k=3, 
        filter=filtros
    )
    
    # Ensamblaje del contexto (Punto 5 de la tarjeta de Trello)
    # Unimos los fragmentos encontrados en una sola cadena de texto estructurada
    contexto_ensamblado = ""
    for i, doc in enumerate(documentos_recuperados, 1):
        contexto_ensamblado += f"--- Fragmento Fuente {i} ---\n"
        contexto_ensamblado += f"{doc.page_content}\n"
        contexto_ensamblado += f"Área Responsable: {doc.metadata['responsable']}\n\n"
        
    return contexto_ensamblado, documentos_recuperados

# Prueba rápida de la capa de recuperación
if __name__ == "__main__":
    from app import iniciar_base_vectorial
    ruta = "data/politica_interna.csv"
    
    db = iniciar_base_vectorial(ruta)
    
    # Simulamos una consulta con filtro de metadatos activado para "Devoluciones"
    pregunta = "¿Cuánto se tarda en reflejar mi dinero?"
    contexto, docs = recuperar_contexto_relevante(db, pregunta, categoria_filtro="Devoluciones")
    
    print("\n🚀 CONTEXTO ENSAMBLADO LISTO PARA ENVIAR AL LLM:")
    print(contexto)
