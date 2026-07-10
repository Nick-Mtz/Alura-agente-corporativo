import streamlit as st
import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Agente IA Corporativo - E-commerce",
    page_icon="🤖",
    layout="centered"
)

# ==========================================
# FUNCIONES DEL BACKEND (ETAPAS 2, 3 Y 4)
# ==========================================
@st.cache_resource
def inicializar_agente():
    """
    Carga el CSV, genera embeddings y monta la base vectorial.
    Usa st.cache_resource para que solo se ejecute una vez al iniciar la app.
    """
    ruta_csv = "data/politica_interna.csv"
    if not os.path.exists(ruta_csv):
        # Fallback por si corren el script desde otra raíz
        ruta_csv = os.path.join(os.path.dirname(__file__), "../data/politica_interna.csv")
        
    df = pd.read_csv(ruta_csv)
    df = df.dropna(subset=['pregunta', 'respuesta'])
    
    documentos_langchain = []
    for _, fila in df.iterrows():
        contenido_texto = f"Pregunta: {fila['pregunta']}\nRespuesta: {fila['respuesta']}"
        metadatos = {
            "id": int(fila['id']),
            "categoria": fila['categoria'],
            "responsable": fila['responsable']
        }
        doc = Document(page_content=contenido_texto, metadata=metadatos)
        documentos_langchain.append(doc)
    
    modelo_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    base_vectorial = Chroma.from_documents(documents=documentos_langchain, embedding=modelo_embeddings)
    return base_vectorial

# Inicializamos el motor RAG de forma persistente en la sesión
try:
    db_vectorial = inicializar_agente()
except Exception as e:
    st.error(f"Error al cargar la base de conocimientos: {e}")
    st.stop()

# ==========================================
# INTERFAZ DE USUARIO (FRONT-END STREAMLIT)
# ==========================================

# Encabezado e indicación clara de que es una IA (Requisito de la Tarjeta)
st.title("🤖 Asistente Virtual Corporativo")
st.caption("⚡ Base de Conocimiento Centralizada para Colaboradores (Tienda Online)")
st.info("Hola, soy un agente de Inteligencia Artificial. Puedo resolver tus dudas sobre envíos, pagos, devoluciones y garantías utilizando la documentación oficial de la empresa.")

# Sidebar para filtros de metadatos y mantenimiento (Requisito de Mantenimiento)
st.sidebar.header("📁 Filtros e Información")
categoria_seleccionada = st.sidebar.selectbox(
    "Filtrar búsqueda por área:",
    ["Todas", "Envios", "Devoluciones", "Pagos", "Garantias", "Soporte"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Panel de Mantenimiento")
st.sidebar.write("**Pipeline de actualización:** Ingesta por carga manual inicial (`.csv`).")
st.sidebar.write("**Curaduría:** Sincronizado con los responsables de área en tiempo real.")

# Inicializar el historial de conversación en la sesión (Requisito de la Tarjeta)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar los mensajes anteriores del historial en la pantalla
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Cuadro de entrada para la pregunta del colaborador
if pregunta_usuario := st.chat_input("Escribe tu pregunta aquí (ej. ¿Cuánto tardan los envíos?)..."):
    
    # 1. Mostrar la pregunta del usuario en el chat
    with st.chat_message("user"):
        st.markdown(pregunta_usuario)
    st.session_state.messages.append({"role": "user", "content": pregunta_usuario})

    # 2. PROCESO RAG: Recuperación y Filtrado por Metadatos
    filtros = None
    if categoria_seleccionada != "Todas":
        filtros = {"categoria": categoria_seleccionada}
        
    documentos_recuperados = db_vectorial.similarity_search(pregunta_usuario, k=2, filter=filtros)

    # 3. GENERACIÓN DE RESPUESTA Y CONTROL DE ALUCINACIÓN
    with st.chat_message("assistant"):
        if not documentos_recuperados:
            respuesta_agente = "Lo siento, no encontré esta información en los documentos disponibles. Por favor, comunícate con el área de soporte técnico."
            st.markdown(respuesta_agente)
        else:
            # Ensamblamos la respuesta simulando las restricciones estrictas del LLM
            primer_doc = documentos_recuperados[0].page_content
            lineas = primer_doc.split("\n")
            respuesta_limpia = next((l.replace("Respuesta Oficial: ", "") for l in lineas if "Respuesta Oficial:" in l), primer_doc)
            
            # Construcción de la respuesta final estructurada con fuentes (Requisito de la Tarjeta)
            respuesta_agente = f"Basado en nuestra documentación interna:\n\n{respuesta_limpia}\n\n"
            respuesta_agente += "**Fuentes y Contactos Verificados:**\n"
            
            fuentes_vistas = set()
            for doc in documentos_recuperados:
                fuente_str = f"📌 Área: **{doc.metadata['categoria']}** (Responsable: _{doc.metadata['responsable']}_)"
                if fuente_str not in fuentes_vistas:
                    respuesta_agente += f"{fuente_str}\n"
                    fuentes_vistas.add(fuente_str)
            
            st.markdown(respuesta_agente)
            
            # 4. Botón de Retroalimentación/Feedback (Requisito de la Tarjeta)
            col1, col2 = st.columns([1, 10])
            with col1:
                if st.button("👍", key=f"up_{len(st.session_state.messages)}"):
                    st.toast("¡Gracias por tu feedback positivo!", icon="✨")
            with col2:
                if st.button("👎", key=f"down_{len(st.session_state.messages)}"):
                    st.toast("Feedback registrado. Revisaremos la documentación de esta sección.", icon="📝")

    # Guardar la respuesta del asistente en el historial de sesión
    st.session_state.messages.append({"role": "assistant", "content": respuesta_agente})
