# main.py
import os
import tempfile
from typing import Any, Dict, List

import streamlit as st
from files import file_uploader, url_uploader
from question import chat_with_doc
from brain import brain
from langchain.docstore.document import Document
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


class CustomSupabaseVectorStore(SupabaseVectorStore):
    '''A custom vector store that uses the match_vectors table instead of the vectors table.'''
    def __init__(self, client: Client, embedding: OpenAIEmbeddings, table_name: str):
        super().__init__(client, embedding, table_name)
    
    def similarity_search(
        self, 
        query: str, 
        table: str = "match_vectors", 
        k: int = 3, 
        threshold: float = 0.5, 
        **kwargs: Any
    ) -> List[Document]:
        vectors = self._embedding.embed_documents([query])
        query_embedding = vectors[0]
        res = self._client.rpc(
            table,
            {
                "query_embedding": query_embedding,
                "match_count": k,
            },
        ).execute()

        match_result = [
            (
                Document(
                    metadata=search.get("metadata", {}),  # type: ignore
                    page_content=search.get("content", ""),
                ),
                search.get("similarity", 0.0),
            )
            for search in res.data
            if search.get("content")
        ]

        documents = [doc for doc, _ in match_result]

        return documents
    

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_store = CustomSupabaseVectorStore(
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
st.markdown("資料を追加すると、その資料の内容にもとづいて答えるようになります。")

# Create a radio button for user to choose between adding knowledge or asking a question
user_choice = st.radio(
    "何をしますか？", ('データを追加', 'チャットする', 'データ削除', "データ確認", "説明"), horizontal=True)

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
elif user_choice == "説明":
    st.sidebar.title("設定")
    st.write("## このアプリでできること")
    st.markdown("""
                データを追加すると、追加されたデータ群の中から近しいと思われるものを参照して質問に回答します。\n
                一般的な質問などには回答できなくなっていますので、そういったものは\n
                - GoogleのBard
                - MicrosoftのBing
                - OpenAIのChatGPT\n
                などに聞いてください。
                """)
    st.write("## データの追加について")
    st.markdown("""
                追加されたデータはいくつかの塊（チャンク）に分割されてデータベースに格納されます。\n
                たとえば前後半の2分割になった場合、質問の内容から後半のデータが参照されたが、肝心の回答は前半にあった場合、質問に対して得られた答えが正確でないということが起こりえます。\n\n
                データ追加後に何分割されたかを確認し、チャンクサイズを調整して再アップロードしたり、チャンクサイズを意識してデータを作成するなどの運用が必要になりそうです。
                """)
    st.write("## 構造")
    st.markdown("""
                このアプリは以下の3つのサービス・ツールを利用しています。
                1. OpenAIのAPI
                2. Supbase
                3. LangChain
                """)
    st.write("### OpenAIのAPI")
    st.markdown("""
                ChatGPTで一世を風靡しているモデルです。チャットの部分に使用しています。\n
                有料ですが高機能かつAPIのため手軽で使いやすいです。
                - gpt-3.5-turbo: $0.002 /1K tokens
                - gpt-4: $0.03 /1K tokens
                """)
    st.write("### Supabase")
    st.markdown("""
                
                """)
    st.write("## ベース")
    st.markdown("""
                ベースはQuivrというオープンソースプロジェクト。\n
                Docker-Composeで立ち上げるスタイルのままWEBアプリ化を断念してこの形式です。
                """)

st.markdown("---\n\n")