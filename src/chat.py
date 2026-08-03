import sys
from search import search_prompt

def main():
    print("Chat iniciado. Digite sua pergunta ou 'sair' para encerrar.")

    while True:
        try:
            question = input("Você: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o chat.")
            break
        
        if not question:
            continue

        if question.lower() == "sair":
            print("Encerrando o chat.")
            break

        try:
            answer = search_prompt(question)
        except Exception as exc:
            print(f"Erro ao obter resposta: {exc}")
            continue

        print(f"Resposta: {answer}")


if __name__ == "__main__":
    main()
