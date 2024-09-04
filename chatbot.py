import os
import re
import time
import streamlit as st
from streamlit_float import *
import streamlit_antd_components as sac

from docx import Document
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import AuthenticationError
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from pymongo import MongoClient
from elasticsearch import Elasticsearch
import warnings
from urllib3.exceptions import InsecureRequestWarning

# Ignorar advertencias de seguridad para conexiones TLS inseguras
warnings.simplefilter('ignore', InsecureRequestWarning)

# Configuración de la conexión a MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["itp_documents"]
collection = db["pdf_texts"]
collection_faq = db["faq"]
collection_keywords = db["keywords"]
collection_categories = db["categories"]
collection_password = db["password"]
collection_settings = db["settings"]  

# Configuración de la conexión a Elasticsearch
es = Elasticsearch(
    ["https://localhost:9200"],
    basic_auth=("elastic", "KcJacMXV1nmiS24cpmoZ"),
    verify_certs=False
)

def load_api_key():
    settings_doc = collection_settings.find_one({"name": "api_key"})
    if settings_doc:
        return settings_doc.get("value")
    return None

def parse_docx(data):
    document = Document(docx=data)
    content = ""
    for para in document.paragraphs:
        content += para.text
    return content

def get_text(doc):
    doc_text = ""
    if ".pdf" in doc.name:
        pdf_reader = PdfReader(doc)
        for each_page in pdf_reader.pages:
            doc_text += each_page.extract_text()
        doc_text += "\n"
    elif ".docx" in doc.name:
        doc_text += parse_docx(data=doc)
    return doc_text

def get_chunks(data):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=2500, chunk_overlap=250, length_function=len
    )
    text_chunks = text_splitter.split_text(data)
    return text_chunks

def store_pdf_text_in_db(pdf_name, text, category):
    if collection.find_one({"name": pdf_name}):
        print(f"Documento {pdf_name} ya existe en la base de datos.")
        return
    document = {
        "name": pdf_name,
        "text": text,
        "category": category
    }
    collection.insert_one(document)

def index_pdf_text_in_elasticsearch(pdf_name, text):
    document = {
        "name": pdf_name,
        "text": text
    }
    es.index(index="itp_documents", body=document)

def load_texts_from_db():
    texts = []
    for document in collection.find():
        texts.append(document["text"])
    return texts

def load_all_texts_from_db():
    texts = []
    for document in collection.find():
        texts.append(f"Documento: {document['name']}\n{document['text']}\n{'-'*80}\n")
    return "\n".join(texts)

def get_vector(chunks):
    return FAISS.from_texts(texts=chunks, embedding=OpenAIEmbeddings())

def get_llm_chain(vectors):
    retriever = vectors.as_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"  # Especifica la clave de salida
    )
    llm_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.7),
        retriever=retriever,
        memory=memory,
        return_source_documents=True  # Devuelve los documentos fuente
    )
    return llm_chain

def update_vectors():
    texts = load_texts_from_db()
    if texts:
        doc_chunks = []
        for text in texts:
            doc_chunks.extend(get_chunks(text))
        vectors = get_vector(doc_chunks)
        st.session_state.llm_chain = get_llm_chain(vectors)
        st.session_state.doc_len = len(texts)

def update_vectors_advanced(pdf_text):
    doc_chunks = get_chunks(pdf_text)
    vectors = get_vector(doc_chunks)
    st.session_state.llm_chain = get_llm_chain(vectors)
    st.session_state.doc_len = 1

def type_effect(text, placeholder):
    full_response = ""
    for chunk in text.split():
        full_response += chunk + " "
        time.sleep(0.05)
        placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
    placeholder.markdown(full_response, unsafe_allow_html=True)

def add_keyword(keyword):
    if not collection_keywords.find_one({"keyword": keyword}):
        collection_keywords.insert_one({"keyword": keyword})

def add_keywords(keywords):
    keywords = [keyword.strip() for keyword in keywords.split(",")]
    for keyword in keywords:
        if not collection_keywords.find_one({"keyword": keyword}):
            collection_keywords.insert_one({"keyword": keyword})

def delete_keyword(keyword):
    collection_keywords.delete_one({"keyword": keyword})

def get_all_keywords():
    keywords = collection_keywords.find()
    return [keyword["keyword"] for keyword in keywords]

def is_related_to_itp(question, additional_keywords=None):
    db_keywords = set(get_all_keywords())
    if additional_keywords:
        all_keywords = db_keywords.union(set(additional_keywords))
    else:
        all_keywords = db_keywords
    return any(keyword.lower() in question.lower() for keyword in all_keywords)

# Funciones para manejar preguntas frecuentes en la base de datos
def add_faq(question, answer, category):
    faq = {
        "question": question,
        "answer": answer,
        "category": category
    }
    collection_faq.insert_one(faq)

    # Indexar la respuesta en el índice vectorial FAISS
    doc_chunks = get_chunks(answer)
    vectors = get_vector(doc_chunks)
    if "vectors" not in st.session_state:
        st.session_state.vectors = vectors
    else:
        st.session_state.vectors.add_texts(doc_chunks)
    st.session_state.llm_chain = get_llm_chain(st.session_state.vectors)


def get_faqs_by_category(category):
    return list(collection_faq.find({"category": category}))

def delete_all_faqs():
    collection_faq.delete_many({})

# Funciones para manejar categorías en la base de datos
def add_category(category):
    if not collection_categories.find_one({"category": category}):
        collection_categories.insert_one({"category": category})

def get_all_categories():
    categories = collection_categories.find()
    return [category["category"] for category in categories]

def highlight_texts(text, highlights):
    for idx, highlight in enumerate(highlights):
        if highlight in text:
            text = text.replace(highlight, f"<mark id='highlight-{idx}'>{highlight}</mark>")
    return text
  
def main():
    st.set_page_config(page_title="Asistente Virtual ITP",
                        # initial_sidebar_state='collapsed',
                        layout="wide",
                        page_icon="static/images/robope.png",
                        menu_items={
                            'Get Help': 'https://itp.edu.co/ITP2022/?page_id=3775',                 
                            }
    )
    
    if "mode" not in st.session_state:
        st.session_state.mode = "basic"

    if "basic_chat_history" not in st.session_state:
        st.session_state.basic_chat_history = []

    if "advanced_chat_history" not in st.session_state:
        st.session_state.advanced_chat_history = []

    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if "highlighted_content" not in st.session_state:
        st.session_state.highlighted_content = load_all_texts_from_db()

    if "llm_chain" not in st.session_state:
        st.session_state.llm_chain = None

    if "doc_len" not in st.session_state:
        st.session_state.doc_len = 0
    
    # Cargar la API key desde la base de datos
    api_key = load_api_key()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.error("No API key found. Please set the API key in the configuration.")


    #  ------------------------------------ sidebar --------------------------------------------------
    itpL = "static/images/itplogo.png"
    logoLet = "static/images/let2.png"
    st.logo(logoLet ,icon_image=itpL)
    
    with st.sidebar:
        select3 = sac.buttons([
            sac.ButtonsItem(label='Nuevo Chat', icon='plus-circle-fill')
        ],index=None, align='center', size='xl', return_index=True)
        
        if select3 == "chatn":
            if st.session_state.mode == "basic":
                st.session_state.basic_chat_history = []
            else:
                st.session_state.advanced_chat_history = []
            st.session_state.chat_history = []
            st.rerun()
        
        options = sac.menu([
            sac.MenuItem('Inicio', icon='house-fill', description='Usuarios'),
            sac.MenuItem('Información', icon='info-circle-fill', description='Usuarios'),
            sac.MenuItem('Configuración', icon='gear-fill', description='Administrador'),
        ], open_all=True, index=0)
     
    # --------------------- CONFIGURACION -----------------------------
    with open("static/css/config.css") as source_des:
                st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

    if options == "Configuración":
        st.title('Configuración')
        multi = ''':gray[Nuestro **software** proporciona una amplia gama de configuraciones que permiten personalizar y gestionar diversos aspectos del sistema de manera eficiente.]'''
        st.markdown(multi)
        
        clave, config = st.columns([2 , 8])
        
        with clave:
            if not st.session_state.password_correct:
                password_input = st.text_input("**Clave de acceso**", type="password")
                if st.button("Ingresar"):
                    stored_password_doc = collection_password.find_one()
                    if stored_password_doc:
                        stored_password = stored_password_doc.get("password")
                        if password_input == stored_password:
                            st.session_state.password_correct = True
                            msg1 = st.toast(':green[Acceso concedido]', icon=':material/thumb_up:')
                            time.sleep(1)
                            msg1.toast(':green[Bienvenido...]', icon=':material/thumb_up:')
                            time.sleep(1)
                            st.rerun()
                        else:
                            msg = st.toast(':red[Verifique nuevamente...]', icon=':material/cancel:')
                            time.sleep(1)
                            msg.toast(':red[Contraseña incorrecta]', icon=':material/cancel:')
                    else:
                        st.error("No se ha configurado ninguna contraseña.")       
            else:
                if st.button("Bloquear secciones"):
                    st.session_state.password_correct = False
                    st.rerun()
                
                    
        with config:
            if st.session_state.password_correct:
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["**Administrar categorías**", "**Administrar PDFs**", "**Administrar FAQs**", "**Palabras clave**", "**Modificar contraseña**", "**Modificar Api Key**"])
                with tab1:
                    st.markdown("**Añadir o eliminar categorías según los PDFs.**")
                    co1, co2 = st.columns([5 , 5])
                    with co1:         
                        new_category = st.text_input("Añadir nueva categoría:")
                        if st.button("Agregar categoría"):
                            if new_category:
                                add_category(new_category)
                                st.toast('Categoría agregada con éxito', icon=':material/check_circle:')
                                # st.success("Categoría agregada con éxito.")
                            else:
                                st.toast('No puedes dejar el espacio vacío', icon=':material/sentiment_dissatisfied:')
                            
                    with co2:
                        categories = get_all_categories()
                        category_to_delete = st.selectbox("Selecciona una categoría para eliminar:", categories, 
                                                          index=None,
                                                          placeholder="Seleccionar...",)
                        if category_to_delete:
                            if st.button("Eliminar categoría"):
                                collection_categories.delete_one({"category": category_to_delete})
                                # st.success("Categoría eliminada con éxito")
                                st.toast(':red[Categoría eliminada con éxito]', icon=':material/delete_forever:')
                                time.sleep(1.5)
                                st.rerun()

                with tab2:
                    co1, co2 = st.columns([5 , 5])
                    with co1:
                        st.markdown("**Añadir PDFs.**")
                        categories = get_all_categories()
                        selected_category = st.selectbox("Selecciona una categoría para los documentos:", categories,
                                                            index=None,
                                                            placeholder="Seleccionar...")
                        
                        if selected_category:
                            doc_upload_placeholder = st.empty()
                            doc = doc_upload_placeholder.file_uploader("Sube el PDF y haz clic en continuar", type=["pdf"])
                            
                            if doc and st.button("Continuar", key="upload_button"):
                                with st.spinner("Procesando..."):
                                    try:
                                        pdf_name = doc.name
                                        doc_text = get_text(doc)
                        
                                        store_pdf_text_in_db(pdf_name, doc_text, selected_category) 
                                        
                                        index_pdf_text_in_elasticsearch(pdf_name, doc_text)  
                                        doc_chunks = get_chunks(doc_text)  
                                        
                                        if "vectors" not in st.session_state:
                                            st.session_state.vectors = get_vector(doc_chunks)  
                                        else:
                                            st.session_state.vectors.add_texts(doc_chunks)  
                                        
                                        st.session_state.llm_chain = get_llm_chain(st.session_state.vectors)
                                        st.session_state.doc_len += 1
                                        
                                        st.toast(':green[Documento cargado exitosamente]', icon=':material/check_circle:')
                                        time.sleep(1)
                                        doc_upload_placeholder.empty()
                                        st.session_state["doc_uploaded"] = True
                                        
                                    except AuthenticationError as e:
                                        st.toast(':red[Error de autenticación. Verifica la API key.]', icon=':material/error:')
                                    except Exception as e:
                                        st.toast(':red[Error al procesar el documento.]', icon=':material/error:')
                                        st.error(f"Detalles del error: {str(e)}") 

                    with co2:
                        st.markdown("**Eliminar PDFs cargados.**")
                        loaded_pdfs = [doc["name"] for doc in collection.find()]
                        selected_pdf = st.selectbox("Selecciona un PDF para eliminar:", loaded_pdfs,
                                                    index=None,
                                                    placeholder="Seleccionar...")

                        if selected_pdf:
                            if st.button("Eliminar PDF"):
                                collection.delete_one({"name": selected_pdf})
                                es.delete_by_query(index="itp_documents", body={"query": {"match": {"name": selected_pdf}}})
                                update_vectors()
                                st.toast(f":red[Documento '{selected_pdf}'eliminado con éxito.]", icon=':material/delete_forever:')
                                time.sleep(1)
                                st.rerun()
                                # st.success(f"Documento '{selected_pdf}' eliminado con éxito.")
                
                        st.markdown("Eliminar todos los PDFs.")   
                        if st.session_state.get("doc_uploaded"):
                            del st.session_state["doc_uploaded"]
                            update_vectors()
                        if st.button(label="Eliminar PDFs", disabled=True):
                            collection.delete_many({})
                            try:
                                es.indices.delete(index="itp_documents", ignore=[400, 404])
                                st.session_state.vectors = None
                                st.session_state.llm_chain = None
                                st.session_state.doc_len = 0
                                st.session_state.chat_history = []
                                st.toast(':red[Todos los PDFs han sido eliminados con éxito]', icon=':material/delete_forever:')
                                time.sleep(1)
                                st.rerun()
                                
                            except Exception as e:
                                st.toast(':orange[Error al eliminar PDFs]', icon=':material/error:')
                                time.sleep(1.5)

                              
                with tab3:
                    co1, co2 = st.columns([5 , 5])
                    with co1:
                        # Formulario para agregar nuevas preguntas
                        st.markdown("**Agregar pregunta frecuente**")
                        question = st.text_input("Pregunta:", key="new_question")
                        answer = st.text_area("Respuesta:", key="new_answer")
                        category = st.selectbox("Categoría:", get_all_categories(), key="new_category")
                        
                        if st.button("Agregar pregunta"):
                            if question and answer and category:
                                add_faq(question, answer, category)
                                st.toast(':green[Pregunta agregada con éxito]', icon=':material/check_circle:')
                                time.sleep(1.5)
                                st.rerun()
                            else:
                                st.toast(':orange[Completa todos los campos antes de agregar]', icon=':material/info:')
                        
                        if st.button("Eliminar todas las preguntas"):
                            delete_all_faqs()
                            st.toast(':red[Preguntas eliminadas]', icon=':material/delete_forever:')
                            time.sleep(1.5)
                            st.rerun()
                        
                    with co2:
                        st.markdown("**Editar o eliminar preguntas frecuentes**")

                        faqs = list(collection_faq.find())

                        selected_faq = st.selectbox("Selecciona una FAQ para editar o eliminar:", 
                                                    [faq["question"] for faq in faqs], 
                                                    key="selected_faq",
                                                    index=None,
                                                    placeholder="Seleccionar una pregunta...")

                        if selected_faq:
                            faq = next(faq for faq in faqs if faq["question"] == selected_faq)
                            edit_question = st.text_area(":blue[Editar la pregunta:]", value=faq["question"], key="edit_question")
                            edit_answer = st.text_area(":blue[Editar la respuesta:]", value=faq["answer"], key="edit_answer")
                            edit_category = st.selectbox(":blue[Editar la categoría:]", get_all_categories(), index=get_all_categories().index(faq["category"]), key="edit_category")

                            if st.button("Guardar cambios"):
                                collection_faq.update_one(
                                    {"_id": faq["_id"]},
                                    {"$set": {"question": edit_question, "answer": edit_answer, "category": edit_category}}
                                )
                                st.toast(':green[Pregunta actualizada con éxito]', icon=':material/check_circle:')
                                time.sleep(1.5)
                                st.rerun()  

                            if st.button("Eliminar pregunta"):
                                collection_faq.delete_one({"_id": faq["_id"]})
                                msg5 = st.toast(':red[Pregunta eliminada con éxito]', icon=':material/delete_forever:')
                                time.sleep(1.5)
                                st.rerun()  
                                
                with tab4:
                    co1, co2 = st.columns([5 , 5])
                    with co1:
                        new_keywords = st.text_area("Nuevas palabras clave (Separadas por comas):")
                        if st.button("Agregar palabras clave"):
                            if new_keywords:
                                add_keywords(new_keywords)
                                st.toast(':green[Palabras clave agregadas con éxito]', icon=':material/check_circle:')
                                time.sleep(1.5)
                                st.rerun()
                            else:
                                st.toast(':orange[Digite palabras clave, antes de presionar el botón]', icon=':material/info:')
                                time.sleep(1.5)
                                
                    with co2:
                        keyword_to_delete = st.selectbox("Selecciona una palabra clave para eliminar:", get_all_keywords(),
                                                        index=None,
                                                        placeholder="Seleccionar una palabra..."
                                                        )
                        if keyword_to_delete:
                            if st.button("Eliminar palabra clave"):
                                delete_keyword(keyword_to_delete)
                                msg7 = st.toast(':red[Palabra clave eliminada con éxito]', icon=':material/delete_forever:')
                                time.sleep(1.5)
                                st.rerun()
                
                with tab5:
                    co1, co2 = st.columns([5 , 5])
                    with co1:
                        st.write("**Modificar contraseña**")
                        new_password = st.text_input("Nueva clave de acceso", type="password", key="new_password")
                        confirm_password = st.text_input("Confirmar nueva clave de acceso", type="password", key="confirm_password")
                        if st.button("Modificar"):
                            if new_password and new_password == confirm_password:
                                collection_password.update_one({}, {"$set": {"password": new_password}}, upsert=True)
                                st.toast(':green[Clave de acceso cambiada con éxito]', icon=':material/check_circle:')
                                time.sleep(1.5)
                                st.session_state.password_correct = False
                                st.rerun()
                            else:
                                st.toast(':orange[Las contraseñas no coinciden o están vacías]', icon=':material/info:')
                
                with tab6:
                    co1, co2 = st.columns([5 , 5])
                    with co1:
                        # Expresión regular para validar API Key de OpenAI
                        api_key_pattern = r"^sk-[a-zA-Z0-9]{32,}$"
                        st.write("**Modificar API Key**")
                        new_api_key = st.text_input("Nueva API Key", type="password", key="new_api_key")
                        if st.button("Modificar API"):
                            if new_api_key:
                                if re.match(api_key_pattern, new_api_key):
                                    collection_settings.update_one({"name": "api_key"}, {"$set": {"value": new_api_key}}, upsert=True)
                                    os.environ["OPENAI_API_KEY"] = new_api_key
                                    st.toast(':green[API Key modificada con éxito]', icon=':material/check_circle:')
                                    time.sleep(1.5)
                                    st.rerun()
                                else:
                                    st.toast(':orange[La API Key ingresada no es válida.]', icon=':material/info:')
                            else:
                                st.toast(':orange[La API Key no puede estar vacía]', icon=':material/info:')
            else:
                st.warning("Ingrese la clave de acceso para desbloquear las secciones.")
                
    #  ------------------------------------ menu --------------------------------------------------
    if options == "Inicio":        
        with open("static/css/main.css") as source_des:
                st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)
        
        selected_mode = sac.segmented(
            items=[
                sac.SegmentedItem(label='Básico', icon='sliders'),
                sac.SegmentedItem(label='Avanzado', icon='layers'),
            ], align='center', radius='xl',color='red', divider=False
        )
        
        chat, pdf = st.columns([5.1 , 4.9])

        with chat:      
            st.image("static/images/robot3.png")
            
            if st.session_state.llm_chain is None:
                with st.spinner("Cargando documentos desde la base de datos..."):
                    try: 
                        update_vectors()
                    except AuthenticationError:
                        st.error("Error de autenticación. Verifica la API key.")

            chat_container = st.container(height=200)
            user_input = st.chat_input("Envía un mensaje al asistente del ITP...", max_chars=200)
            
            #  ------------------------------------ basico --------------------------------------------------

            if st.session_state.mode == "basic":
                st.session_state.chat_history = st.session_state.basic_chat_history

                if not user_input:
                    categories = get_all_categories()
                    faq_type = chat_container.selectbox(
                        "**Selecciona una categoría de preguntas frecuentes:**",
                        categories,
                        placeholder= "Selecciona una categoria de preguntas frecuentes....",
                        index=None,
                        label_visibility="collapsed"
                    )
                    if faq_type != "Selecciona una categoria de preguntas frecuentes....":
                        faqs = get_faqs_by_category(faq_type)
                        num_faqs = len(faqs)
                        if num_faqs > 0:
                            num_columns = 3
                            faqs_per_column = (num_faqs + num_columns - 1) // num_columns
                            columns = chat_container.columns(num_columns, gap="large")

                            for i in range(num_columns):
                                with columns[i]:
                                    for faq in faqs[i * faqs_per_column:(i + 1) * faqs_per_column]:
                                        if st.button(faq["question"]):
                                            with chat_container.chat_message("assistant", avatar="static/images/asis2.png"):
                                                # Buscar fragmentos relevantes en los documentos
                                                try:
                                                    response = st.session_state.llm_chain({"question": faq["question"]})
                                                    source_documents = response["source_documents"]

                                                    highlighted_texts = [doc.page_content for doc in source_documents[:2]]
                                                    all_texts = load_all_texts_from_db()
                                                    highlighted_content = highlight_texts(all_texts, highlighted_texts)
                                                    st.session_state.highlighted_content = highlighted_content

                                                    links = " ".join([f"<span style='margin-right: 10px;'><a href='#highlight-{idx}' id='highlight-link-{idx}'>{idx+1}</a></span>" for idx in range(len(highlighted_texts))])
                                                    bot_response_with_links = f"{faq['answer']}<br><br>{links}"

                                                    message_placeholder = st.empty()
                                                    type_effect(bot_response_with_links, message_placeholder)
                                                except AuthenticationError:
                                                    st.error("Error de autenticación. Verifica la API key.")
                                                except Exception as e:
                                                    st.error("Error al procesar la solicitud.")
                                                

                if user_input and st.session_state.llm_chain:
                    if is_related_to_itp(user_input):
                        try:
                            response = st.session_state.llm_chain({"question": user_input})
                            bot_response = response["answer"]
                            source_documents = response["source_documents"]

                            highlighted_texts = [doc.page_content for doc in source_documents[:2]]
                            all_texts = load_all_texts_from_db()
                            highlighted_content = highlight_texts(all_texts, highlighted_texts)
                            st.session_state.highlighted_content = highlighted_content

                            links = " ".join([f"<span style='margin-right: 10px;'><a href='#highlight-{idx}' id='highlight-link-{idx}'>{idx+1}</a></span>" for idx in range(len(highlighted_texts))])
                            bot_response_with_links = f"{bot_response}<br><br>{links}"

                            st.session_state.chat_history.append({"role": "user", "content": user_input})
                            st.session_state.chat_history.append({"role": "assistant", "content": bot_response_with_links})

                            for idx, msg in enumerate(st.session_state.chat_history[:-1]):
                                if msg["role"] == "user":
                                    with chat_container.chat_message("user", avatar="static/images/userlog.png"):
                                        st.write(msg["content"])
                                else:
                                    with chat_container.chat_message("assistant", avatar="static/images/asis2.png"):
                                        st.markdown(msg["content"], unsafe_allow_html=True)

                            last_message = st.session_state.chat_history[-1]
                            if last_message["role"] == "assistant":
                                with chat_container.chat_message("assistant", avatar="static/images/asis2.png"):
                                    message_placeholder = st.empty()
                                    type_effect(last_message["content"], message_placeholder)
                        except AuthenticationError:
                            st.error("Error de autenticación. Verifica la API key.")
                        except Exception as e:
                            st.error("Error al procesar la solicitud.")
                    else:
                        with chat_container.chat_message("assistant", avatar="static/images/asis2.png"):
                            st.write("La pregunta no está relacionada con el Instituto Tecnológico del Putumayo.")
                elif user_input and st.session_state.llm_chain is None:
                    chat_container.error("Cargue archivos y haga clic en continuar antes de hacer preguntas.")

            #  ------------------------------------ avanzado --------------------------------------------------

            elif st.session_state.mode == "advanced":
                st.session_state.chat_history = st.session_state.advanced_chat_history

                if "pdf_texts" not in st.session_state:
                    st.session_state.pdf_texts = {doc["name"]: {"text": doc["text"], "category": doc["category"]} for doc in collection.find()}

                if "selected_category" not in st.session_state:
                    st.session_state.selected_category = None

                if not user_input:
                    categories = get_all_categories()
                    selected_category = chat_container.selectbox("Selecciona una categoría", 
                                                                 categories,
                                                                 placeholder= "Selecciona una categoria de preguntas frecuentes....",
                                                                 index=None,
                                                                 label_visibility="collapsed")

                    if selected_category != "Selecciona una categoría..." and selected_category != st.session_state.selected_category:
                        st.session_state.selected_category = selected_category
                        pdfs_in_category = [doc for doc in collection.find({"category": selected_category})]
                        if pdfs_in_category:
                            pdf_text = pdfs_in_category[0]["text"]
                            st.session_state.highlighted_content = pdf_text
                            try:
                                update_vectors_advanced(pdf_text)
                            except AuthenticationError:
                                st.error("Error de autenticación. Verifica la API key.")
                            except Exception as e:
                                st.error("Error al actualizar los vectores.")
                    if st.session_state.selected_category:
                        faqs = get_faqs_by_category(st.session_state.selected_category)
                        num_faqs = len(faqs)
                        if num_faqs > 0:
                            num_columns = 3
                            faqs_per_column = (num_faqs + num_columns - 1) // num_columns
                            columns = chat_container.columns(num_columns, gap="large")

                            for i in range(num_columns):
                                with columns[i]:
                                    for faq in faqs[i * faqs_per_column:(i + 1) * faqs_per_column]:
                                        if st.button(faq["question"]):
                                            with chat_container.chat_message("assistant", avatar="static/images/asis2.png"):
                                                # Buscar fragmentos relevantes en los documentos
                                                try: 
                                                    response = st.session_state.llm_chain({"question": faq["question"]})
                                                    source_documents = response["source_documents"]

                                                    highlighted_texts = [doc.page_content for doc in source_documents[:2]]
                                                    all_texts = load_all_texts_from_db()
                                                    highlighted_content = highlight_texts(all_texts, highlighted_texts)
                                                    st.session_state.highlighted_content = highlighted_content

                                                    links = " ".join([f"<span style='margin-right: 10px;'><a href='#highlight-{idx}' id='highlight-link-{idx}'>{idx+1}</a></span>" for idx in range(len(highlighted_texts))])
                                                    bot_response_with_links = f"{faq['answer']}<br><br>{links}"

                                                    message_placeholder = st.empty()
                                                    type_effect(bot_response_with_links, message_placeholder)
                                                except AuthenticationError:
                                                    st.error("Error de autenticación. Verifica la API key.")
                                                except Exception as e:
                                                    st.error("Error al procesar la solicitud.")

                if user_input and st.session_state.llm_chain:
                    # Reset highlighted content to the current selected category content before each new query
                    pdfs_in_category = [doc for doc in collection.find({"category": st.session_state.selected_category})]
                    if pdfs_in_category:
                        st.session_state.highlighted_content = pdfs_in_category[0]["text"]

                    try:
                        response = st.session_state.llm_chain({"question": user_input})
                        bot_response = response["answer"]
                        source_documents = response["source_documents"]

                        highlighted_texts = [doc.page_content for doc in source_documents[:2]]
                        highlighted_content = highlight_texts(st.session_state.highlighted_content, highlighted_texts)
                        st.session_state.highlighted_content = highlighted_content

                        links = " ".join([f"<span style='margin-right: 10px;'><a href='#highlight-{idx}' id='highlight-link-{idx}'>{idx+1}</a></span>" for idx in range(len(highlighted_texts))])
                        bot_response_with_links = f"{bot_response}<br><br>{links}"

                        st.session_state.chat_history.append({"role": "user", "content": user_input})
                        st.session_state.chat_history.append({"role": "assistant", "content": bot_response_with_links})

                        for idx, msg in enumerate(st.session_state.chat_history[:-1]):
                            if msg["role"] == "user":
                                with chat_container.chat_message("user", avatar="static/images/userlog.png"):
                                    st.write(msg["content"])
                            else:
                                with chat_container.chat_message("assistant", avatar="static/images/asis2.png"):
                                    st.markdown(msg["content"], unsafe_allow_html=True)

                        last_message = st.session_state.chat_history[-1]
                        if last_message["role"] == "assistant":
                            with chat_container.chat_message("assistant", avatar="static/images/asis2.png"):
                                message_placeholder = st.empty()
                                type_effect(last_message["content"], message_placeholder)
                    except AuthenticationError:
                        st.error("Error de autenticación. Verifica la API key.")
                    except Exception as e:
                        st.error("Error al procesar la solicitud.")
                elif user_input and st.session_state.llm_chain is None:
                        chat_container.error("Selecciona una categoría antes de hacer preguntas.")

            # ------------------------ Cambiar modos -------------------------------        
            if selected_mode == "Básico" and st.session_state.mode != "basic":
                st.session_state.advanced_chat_history = st.session_state.chat_history
                st.session_state.mode = "basic"
                try:
                    update_vectors()
                except AuthenticationError:
                    st.error("Error de autenticación. Verifica la API key.")
                except Exception as e:
                    st.error("Error al actualizar los vectores.")
                st.session_state.highlighted_content = load_all_texts_from_db()
                st.session_state.chat_history = st.session_state.basic_chat_history
                st.rerun()
            if selected_mode == "Avanzado" and st.session_state.mode != "advanced":
                st.session_state.basic_chat_history = st.session_state.chat_history
                st.session_state.mode = "advanced"
                st.session_state.chat_history = st.session_state.advanced_chat_history
                st.rerun()

        #  ------------------------------------ PDF --------------------------------------------------
        with pdf:
            st.subheader("Documentos PDF 📄")
            st.markdown(f"<div style='max-height: 80vh; overflow-y: auto;border-radius: 10px;border-bottom: 3px solid #b5b5b5;padding: 6px;'>{st.session_state.highlighted_content}</div>", unsafe_allow_html=True)

    if options == "Información":
        # Opciones de color
        background_color = "#f9f9f9"
        border_color = "#ccc"
        text_color = "#000"
        header_color = "#333"
        box_shadow = "0 4px 8px 0 rgba(0, 0, 0, 0.2)"

        st.title('Información básica del Asistente Virtual 🤖')

        # Secciones de información
        co1, co2 = st.columns([1, 1])

        with co1:
            multi = f'''
            <div style="max-height: 76vh; overflow-y: auto; padding: 20px; border-radius: 20px; border: 2px solid {border_color}; background-color: {background_color}; color: {text_color}; box-shadow: {box_shadow};">
                <h3 style="color: {header_color};">Modo Básico</h3>
                <p><strong>Descripción:</strong></p>
                <p>El modo básico ofrece una interfaz simple y accesible para todos los usuarios. Este modo está diseñado pensando en la facilidad de uso y la rapidez de acceso a la información. Ideal para obtener respuestas inmediatas y acceder a recursos generales sin complicaciones.</p>
                <p><strong>Funcionalidades:</strong></p>
                <ul>
                    <li><strong>Chat:</strong> Envía mensajes para recibir respuestas rápidas y precisas. Nuestra inteligencia artificial está siempre disponible para resolver tus dudas en tiempo real, garantizando una experiencia fluida y eficaz.</li>
                    <li><strong>Preguntas Frecuentes:</strong> Accede a una base de datos de respuestas a las consultas más comunes. Este recurso te permitirá encontrar soluciones rápidas a problemas frecuentes sin necesidad de esperar.</li>
                    <li><strong>Documentos PDF:</strong> Visualiza todos los documentos disponibles de manera sencilla. Nuestra biblioteca de recursos está organizada para que encuentres fácilmente la información que necesitas.</li>
                </ul>
            </div>
            '''
            st.markdown(multi, unsafe_allow_html=True)

        with co2:
            multi = f'''
            <div style="max-height: 76vh; overflow-y: auto; padding: 20px; border-radius: 20px; border: 2px solid {border_color}; background-color: {background_color}; color: {text_color}; box-shadow: {box_shadow};">
                <h3 style="color: {header_color};">Modo Avanzado</h3>
                <p><strong>Descripción:</strong></p>
                <p>El modo avanzado está diseñado para usuarios que necesitan un control más detallado y preciso sobre la información que consultan. Este modo permite la selección de categorías y documentos específicos, proporcionando una experiencia más personalizada y eficiente.</p>
                <p><strong>Funcionalidades:</strong></p>
                <ul>
                    <li><strong>Chat:</strong> Envía mensajes y recibe respuestas basadas en documentos específicos. Ideal para usuarios que requieren respuestas detalladas y contextualizadas según sus necesidades particulares.</li>
                    <li><strong>Selección de Categorías:</strong> Elige una categoría para centrar tus consultas en documentos específicos. Esto facilita la búsqueda y el acceso a la información más relevante y precisa.</li>
                    <li><strong>Documentos PDF:</strong> Visualiza el documento específico de la categoría seleccionada. Accede rápidamente a la documentación precisa y relevante para tus consultas. Esta función permite a los usuarios verificar y profundizar aun mas sobre la información directamente desde la fuente.</li>
                </ul>
            </div>
            '''
            st.markdown(multi, unsafe_allow_html=True)

        st.divider() 
        
        equipo = [
            {"nombre": "Yan Carlos Cuaran", "correo": "yancuaran2020@itp.edu.co", "rol": "Ingeniero de Sistemas", "imagen": "static/images/Yan.png"},
            {"nombre": "Maicol Jossa Campaña", "correo": "maicoljossa2020@itp.edu.co", "rol": "Ingeniero de Sistemas", "imagen": "static/images/maicol.png"},
            {"nombre": "Jhoni Ceron", "correo": "jceronch@itp.edu.co", "rol": "Doctor", "imagen": "static/images/jhoni.jpg"}
        ]

        st.subheader("Equipo de Desarrollo y Asesor")

        # Crear la tabla
        for miembro in equipo:
            col1, col2 = st.columns([0.5, 4.5])
            with col1:
                st.image(miembro["imagen"])
                
            with col2:
                st.markdown(f"""
                **{miembro['nombre']}**  
                *{miembro['correo']}*  
                {miembro['rol']}
                """)

         # st.caption("A caption with _italics_ :blue[colors] and emojis :sunglasses:")

if __name__ == "__main__":
    main()