import os

from langchain import OpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.llms import GPT4All
from langchain.memory import ConversationBufferMemory


class Config:
    obsidian_root: str
    local_model: str
    max_context_size: int
    chunk_content_size: int
    chunk_overlap: int
    generated_content_size: int
    temperature: float
    top_p: float
    top_k: int

    def get_embeddings(self):
        embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(model_name=embeddings_model)

    def get_llm(self, callbacks):
        return GPT4All(
            callbacks=callbacks,
            model=self.local_model,
            n_ctx=self.max_context_size,
            n_predict=self.generated_content_size,
            temp=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            verbose=False,
        )

    def get_chain(self, llm, db):
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            verbose=False,
        )


class OpenAiConfig(Config):

    def __init__(self, obsidian_root: str):
        os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_ACCESS_TOKEN']

        self.obsidian_root = obsidian_root
        self.local_model = f"{os.environ['HOME']}/Library/Application Support/nomic.ai/GPT4All/openai-gpt.bin"
        self.max_context_size = 4096
        self.chunk_content_size = 512
        self.chunk_overlap = 128
        self.generated_content_size = 768
        self.temperature = 0.7
        self.top_p = 0.85
        self.top_k = 5

    def get_embeddings(self):
        return OpenAIEmbeddings(model="text-embedding-ada-002")

    def get_llm(self, callbacks):
        return OpenAI(
            callbacks=callbacks,
            model="text-davinci-003",
            max_tokens=self.generated_content_size,
            temperature=self.temperature,
            # top_p=self.top_p,
            verbose=False
        )


class OpenAiChatConfig(Config):

    def __init__(self, obsidian_root: str):
        os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_ACCESS_TOKEN']

        self.obsidian_root = obsidian_root
        self.local_model = f"{os.environ['HOME']}/Library/Application Support/nomic.ai/GPT4All/openai-gpt.bin"
        self.max_context_size = 8192
        self.chunk_content_size = 512
        self.chunk_overlap = 128
        self.generated_content_size = 768
        self.temperature = 0.7
        self.top_p = 0.85
        self.top_k = 5

    def get_embeddings(self):
        return OpenAIEmbeddings(model="text-embedding-ada-002")

    def get_llm(self, callbacks):
        return ChatOpenAI(
            callbacks=callbacks,
            model="gpt-4-0613",
            max_tokens=self.generated_content_size,
            temperature=self.temperature,
            # top_p=self.top_p,
            verbose=False
        )

    def get_chain(self, llm, db):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="result")
        return ConversationalRetrievalChain.from_llm(
            llm,
            db.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            chain_type="stuff",
            return_source_documents=True,
            output_key="result",
            get_chat_history=lambda h: h,
            verbose=False,
        )


# This is not yet functinoal, it will not work for you.
class Mpt7bChatConfig(Config):

    def __init__(self, obsidian_root: str):
        self.obsidian_root = obsidian_root
        self.local_model = f"{os.environ['HOME']}/Library/Application Support/nomic.ai/GPT4All/ggml-mpt-7b-chat.bin"
        self.max_context_size = 2048
        self.chunk_content_size = 384
        self.chunk_overlap = 64
        self.generated_content_size = 512
        self.temperature = 0
        self.top_p = 0.85
        self.top_k = 5

    def get_chain(self, llm, db):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="result")
        return ConversationalRetrievalChain.from_llm(
            llm,
            db.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            chain_type="stuff",
            return_source_documents=True,
            output_key="result",
            get_chat_history=lambda h: h,
            verbose=False,
        )


class Mpt7bInstructConfig(Config):

    def __init__(self, obsidian_root: str):
        self.obsidian_root = obsidian_root
        self.local_model = f"{os.environ['HOME']}/Library/Application Support/nomic.ai/GPT4All/ggml-mpt-7b-instruct.bin"
        self.max_context_size = 2048
        self.chunk_content_size = 384
        self.chunk_overlap = 64
        self.generated_content_size = 512
        self.temperature = 0
        self.top_p = 0.85
        self.top_k = 5


class NousHermes13bConfig(Config):

    def __init__(self, obsidian_root: str):
        self.obsidian_root = obsidian_root
        self.local_model = f"{os.environ['HOME']}/Library/Application Support/nomic.ai/GPT4All/nous-hermes-13b.ggmlv3.q4_0.bin"
        self.max_context_size = 2048
        self.chunk_content_size = 384
        self.chunk_overlap = 64
        self.generated_content_size = 512
        self.temperature = 0
        self.top_p = 0.85
        self.top_k = 5


class NousGpt4Vicuna13bConfig(Config):

    def __init__(self, obsidian_root: str):
        self.obsidian_root = obsidian_root
        self.local_model = f"{os.environ['HOME']}/Library/Application Support/nomic.ai/GPT4All/ggml-nous-gpt4-vicuna-13b.bin"
        self.max_context_size = 2048
        self.chunk_content_size = 384
        self.chunk_overlap = 64
        self.generated_content_size = 512
        self.temperature = 0
        self.top_p = 0.85
        self.top_k = 5


class StableVicuna13bConfig(Config):

    def __init__(self, obsidian_root: str):
        self.obsidian_root = obsidian_root
        self.local_model = f"{os.environ['HOME']}/Library/Application Support/nomic.ai/GPT4All/ggml-stable-vicuna-13B.q4_2.bin"
        self.max_context_size = 2048
        self.chunk_content_size = 384
        self.chunk_overlap = 64
        self.generated_content_size = 512
        self.temperature = 0
        self.top_p = 0.85
        self.top_k = 5


class Gpt4alll13bSnoozy(Config):

    def __init__(self, obsidian_root: str):
        self.obsidian_root = obsidian_root
        self.local_model = f"{os.environ['HOME']}/Library/Application Support/nomic.ai/GPT4All/ggml-gpt4all-l13b-snoozy.bin"
        self.max_context_size = 2048
        self.chunk_content_size = 384
        self.chunk_overlap = 64
        self.generated_content_size = 512
        self.temperature = 0
        self.top_p = 0.85
        self.top_k = 5
