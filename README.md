# Memory Agent

The Memory Agent is a Python application designed to help users store and retrieve memories using a vector database and a language model. It allows for the extraction of concise facts from user input and provides responses based on stored memories.

## Installation

To set up the Memory Agent, you need to install the required dependencies. You can do this using pip. First, ensure you have Python and pip installed on your machine. Then, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

The required packages are:
- `ollama`: A library for interacting with language models.
- `chromadb`: A library for managing vector databases.
- `sentence-transformers`: A library for sentence embeddings.

## Usage

1. Clone the repository or download the source code.
2. Navigate to the project directory.
3. Run the application using the following command:

```bash
python src/app.py
```

4. Follow the on-screen instructions to interact with the Memory Agent.

## Features

- Save and retrieve memories based on user input.
- Extract concise facts from user messages.
- Clear all memories or reset memories for a specific user.

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
