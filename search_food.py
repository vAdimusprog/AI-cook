from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.retrievers import BM25Retriever
# from langchain_community.retrievers  import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
import os
from dotenv import load_dotenv


class RAGSystem:
    def __init__(self, text):
        load_dotenv()
        self.client = OpenAI(
            base_url="https://foundation-models.api.cloud.ru/v1",
            api_key=os.getenv("GIGA")
        )
        self.text_processor = text
        self.text_processor.procces()

    def ask_question(self, question):
        # 1. Находим релевантные документы
        relevant_docs = self.text_processor.search_documents(question, k=3)

        # 2. Формируем контекст
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # 3. Отправляем запрос к модели
        completion = self.client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507",
            messages=[
                {
                    "role": "system",
                    "content": f"""Дай ответ по контексту. Если вопроса нет в контексте, напиши об этом"""
                },
                {
                    "role": "user",
                    "content": f"Контекст: {context}\n\nВопрос: {question}"
                }
            ]
        )

        return completion.choices[0].message.content


# Убирает предупреждение
os.environ["USER_AGENT"] = "RAG_Bot/1.0"


class Text():
    def __init__(self, name):
        self.name = name
        self.docs = []
        self.vector_store = None
        self.retriever = None

    def procces(self):
        loader = TextLoader(self.name, encoding="utf-8")
        documents = loader.load()

        # Если файл большой - разделить на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        docs = text_splitter.split_documents(documents)
        self.docs = docs
        self.vector_store = self.make_vector(docs)
        self.create_retriever()

    def make_vector(self, docs):
        model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}

        embedding = HuggingFaceEmbeddings(model_name=model_name,
                                          model_kwargs=model_kwargs,
                                          encode_kwargs=encode_kwargs)

        vector_store = self.create_or_load_vector_store(docs, embedding)

        return vector_store

    def create_or_load_vector_store(self, docs, embedding, save_path="faiss_index"):
        if os.path.exists(f"{save_path}/index.faiss") and os.path.exists(f"{save_path}/index.pkl"):
            # Загружаем существующую БД
            vector_store = FAISS.load_local(save_path, embedding, allow_dangerous_deserialization=True)
            print("Векторная БД загружена из файла")
        else:
            # Создаем новую БД
            vector_store = FAISS.from_documents(docs, embedding=embedding)
            # Сохраняем для будущего использования
            vector_store.save_local(save_path)
            print("Векторная БД создана и сохранена")

        return vector_store

    def create_retriever(self, k=5):
        """Создает retriever для поиска документов"""

        if self.vector_store is None:
            raise ValueError("Сначала выполните process() для создания векторного хранилища")

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        return self.retriever

    def get_retriever(self, k=5):
        """Возвращает retriever (создает если нужно)"""
        if self.retriever is None:
            self.create_retriever(k)
        return self.retriever

    def search_documents(self, query, k=5):
        """Поиск документов по запросу"""
        retriever = self.get_retriever(k)
        return  retriever.invoke(query)

    def get_docs(self):
        return self.docs

    def get_vector_store(self):
        return self.vector_store

txt = Text("saved_txt.txt")
txt.procces()
print(len(txt.get_docs()))
#
#
# # Убедитесь что .env файл в правильной папке
load_dotenv()
#
#
rag = RAGSystem(txt)
answer = rag.ask_question("Как приготовить жареную рыбу и что для этого нужно?")
print(answer)
