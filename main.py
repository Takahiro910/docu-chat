# main.py
import os
import tempfile

import streamlit as st
from files import file_uploader, url_uploader
from question import chat_with_doc
from brain import brain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from supabase import Client, create_client
from explorer import view_document


supabase_url = st.secrets.supabase_url
supabase_key = st.secrets.supabase_service_key
openai_api_key = st.secrets.openai_api_key
anthropic_api_key = st.secrets.anthropic_api_key
supabase: Client = create_client(supabase_url, supabase_key)
self_hosted = st.secrets.self_hosted

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_store = SupabaseVectorStore(
    supabase, embeddings, table_name="vectors")
models = ["gpt-3.5-turbo", "gpt-4"]
if anthropic_api_key:
    models += ["claude-v1", "claude-v1.3",
               "claude-instant-v1-100k", "claude-instant-v1.1-100k"]

# Set the theme
st.set_page_config(
    page_title="Akasha",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("🧠 外付け脳（Proto）")
st.markdown("資料を追加すると、その資料の内容についても答えられるようになります。")

# Create a radio button for user to choose between adding knowledge or asking a question
user_choice = st.radio(
    "何する？", ('データを追加', 'チャットする', 'データ削除', "データ確認"), horizontal=True)

st.markdown("---\n\n")

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state['model'] = "gpt-3.5-turbo"
if 'temperature' not in st.session_state:
    st.session_state['temperature'] = 0.1
if 'chunk_size' not in st.session_state:
    st.session_state['chunk_size'] = 500
if 'chunk_overlap' not in st.session_state:
    st.session_state['chunk_overlap'] = 0
if 'max_tokens' not in st.session_state:
    st.session_state['max_tokens'] = 1024


if user_choice == 'データを追加':
    # Display chunk size and overlap selection only when adding knowledge
    st.sidebar.title("設定")
    st.sidebar.markdown(
        "Choose your chunk size and overlap for adding knowledge.")
    st.session_state['chunk_size'] = st.sidebar.slider(
        "Select Chunk Size", 100, 1000, st.session_state['chunk_size'], 50)
    st.session_state['chunk_overlap'] = st.sidebar.slider(
        "Select Chunk Overlap", 0, 100, st.session_state['chunk_overlap'], 10)
    
    # Create two columns for the file uploader and URL uploader
    col1, col2 = st.columns(2)
    
    with col1:
        file_uploader(supabase, vector_store)
    with col2:
        url_uploader(supabase, vector_store)
elif user_choice == 'チャットする':
    # Display model and temperature selection only when asking questions
    st.sidebar.title("設定")
    st.sidebar.markdown(
        "モデル、温度、トークン上限を選択してください。")

    st.session_state['model'] = st.sidebar.selectbox(
    "Select Model", models, index=(models).index(st.session_state['model']))

    st.session_state['temperature'] = st.sidebar.slider(
        "Select Temperature", 0.0, 1.0, st.session_state['temperature'], 0.1)

    st.session_state['max_tokens'] = st.sidebar.slider(
        "Select Max Tokens", 256, 4096, st.session_state['max_tokens'], 1)
    
    chat_with_doc(st.session_state['model'], vector_store, stats_db=supabase)
elif user_choice == 'データ削除':
    st.sidebar.title("設定")
    brain(supabase)
elif user_choice == 'データ確認':
    st.sidebar.title("設定")
    view_document(supabase)

st.markdown("---\n\n")