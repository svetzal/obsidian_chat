import os
import shutil
import sys

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from colorama import Fore, Back, Style

from obsidian_splitting_loader import ObsidianSplittingLoader

os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_ACCESS_TOKEN']


def main(obsidian_root):
    # Remove the db directory between runs to avoid loading old data if it exists
    if os.path.exists("db"):
        shutil.rmtree("db")

    callbacks = [StreamingStdOutCallbackHandler()]

    ###############################
    # mpt-7b-instruct
    #
    local_model = f"{os.environ['HOME']}/Library/Application Support/nomic.ai/GPT4All/ggml-mpt-7b-instruct.bin"
    max_context_size = 2048
    chunk_content_size = 384
    generated_content_size = 512
    chunk_overlap = 64
    llm = GPT4All(
        callbacks=callbacks,
        model=local_model,
        n_ctx=max_context_size,
        n_predict=generated_content_size,
        temp=0,
        top_p=0.85,
        top_k=5,
        # repeat_last_n=128,
        # repeat_penalty=2,
        verbose=False
    )
    embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_content_size, chunk_overlap=chunk_overlap)
    #
    ###############################

    # Super short response?
    # local_model = "/Users/svetzal/Library/Application Support/nomic.ai/GPT4All/ggml-gpt4all-l13b-snoozy.bin"

    # Great response!
    # local_model = "/Users/svetzal/Library/Application Support/nomic.ai/GPT4All/nous-hermes-13b.ggmlv3.q4_0.bin"

    # ???
    # local_model = "/Users/svetzal/Library/Application Support/nomic.ai/GPT4All/ggml-nous-gpt4-vicuna-13b.bin"

    # ???
    # local_model = "/Users/svetzal/Library/Application Support/nomic.ai/GPT4All/ggml-stable-vicuna-13B.q4_2.bin"

    # llm = OpenAI(
    #     callbacks=callbacks,
    #     # model="text-curie-001",
    #     max_tokens=generated_content_size,
    #     temperature=0.8,
    #     top_p=1,
    #     verbose=False
    # )

    print(Fore.BLUE + "Loading vault..." + Style.RESET_ALL)

    # get the obsidian_root from the first command-line argument
    obsidian_docs = ObsidianSplittingLoader(obsidian_root, splitter).load()
    db = Chroma.from_documents(obsidian_docs, embeddings, collection_name="personal_vault", persist_directory="db")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        verbose=False,
    )

    print(Fore.BLUE + "Ready." + Style.RESET_ALL)

    while True:
        question = input("> ")

        qa_response = qa(question)
        print("\n" + Fore.GREEN + "References:" + Style.RESET_ALL)
        [print(Fore.GREEN + d.metadata['source'] + Style.RESET_ALL) for d in qa_response['source_documents']]


if __name__ == '__main__':
    obsidian_root = sys.argv[1]
    print(obsidian_root)
    main(obsidian_root)
