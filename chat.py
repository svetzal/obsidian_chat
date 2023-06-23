from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from config import Mpt7bInstructConfig
from obsidian_splitting_loader import ObsidianSplittingLoader


class Chat:

    def __init__(self, config: Mpt7bInstructConfig):
        self.config = config
        self.callbacks = [StreamingStdOutCallbackHandler()]
        self.llm = config.get_llm(self.callbacks)
        self.embeddings = config.get_embeddings()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.config.chunk_content_size,
                                                       chunk_overlap=self.config.chunk_overlap)
        obsidian_docs = ObsidianSplittingLoader(self.config.obsidian_root, self.splitter).load()
        self.db = Chroma.from_documents(obsidian_docs, self.embeddings, collection_name="personal_vault",
                                        persist_directory="db")
        self.qa = config.get_chain(self.llm, self.db)

    def ask(self, question: str):
        qa_response = self.qa(question)
        return qa_response
