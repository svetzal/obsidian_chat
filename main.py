import os
import shutil
import sys

from colorama import Fore, Style

from chat import Chat
from config import Mpt7bInstructConfig


def main(obsidian_root):
    remove_db()

    config = Mpt7bInstructConfig(obsidian_root)

    print(Fore.BLUE + "Initializing..." + Style.RESET_ALL)

    chat = Chat(config)

    print(Fore.BLUE + "Ready." + Style.RESET_ALL)

    while True:
        question = input("> ")

        response = chat.ask(question)
        print(response['result'])
        # print(response['answer']) # working through chat
        print("\n" + Fore.GREEN + "References:" + Style.RESET_ALL)
        [print(Fore.GREEN + d.metadata['source'] + Style.RESET_ALL) for d in response['source_documents']]


def remove_db():
    # Remove the db directory between runs to avoid loading old data if it exists
    if os.path.exists("db"):
        shutil.rmtree("db")


if __name__ == '__main__':
    obsidian_root = sys.argv[1]
    print(obsidian_root)
    main(obsidian_root)
