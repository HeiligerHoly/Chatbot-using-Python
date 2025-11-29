"""Command line interface for chatting with the trained model."""

from chatbot_core import Chatbot


def main() -> None:
    bot = Chatbot()
    print("Chatbot gestartet. Tippe 'exit' oder 'quit' zum Beenden.")

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAuf Wiedersehen!")
            break

        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            print("Auf Wiedersehen!")
            break

        response = bot.chatbot_response(message)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()
