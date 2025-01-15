# Study Assistant v2 🎓

An intelligent study companion that uses AI to process study materials, generate comprehensive memory maps, and provide interactive Q&A capabilities. Built with LangChain and OpenAI's GPT-4.

## Features 🌟

- **Document Processing**
  - PDF text extraction
  - Handwritten notes OCR (Optical Character Recognition)
  - Image preprocessing for improved text recognition
  - Support for multiple file formats (PDF, JPG, PNG)

- **Memory Map Generation**
  - Creates structured knowledge representations
  - Identifies core concepts and relationships
  - Highlights key study points
  - Tracks confidence levels in information

- **Interactive Q&A System**
  - Context-aware responses
  - Source citation for answers
  - Conversation memory for coherent dialogue
  - Maximum Marginal Relevance search for diverse responses

## Technologies Used 🛠️

- Python 3.9+
- OpenAI GPT-4
- LangChain
- Streamlit
- PyTesseract
- OpenCV
- FAISS Vector Store
- PDFPlumber

## Prerequisites 📋

- Python 3.9 or higher
- OpenAI API key
- Tesseract OCR installed (for handwritten notes)

## Installation 🔧

1. Clone the repository:
```bash
git clone https://github.com/yourusername/study_assistant_v2.git
cd study_assistant_v2
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
TESSERACT_PATH=/path/to/tesseract  # Optional: Only needed if Tesseract is not in PATH
```

## Usage 🚀

1. Start the application:
```bash
streamlit run app.py
```

2. Enter your OpenAI API key in the web interface

3. Upload study materials (PDFs or images)

4. Generate memory maps by entering topics

5. Ask questions about your study materials

## Project Structure 📁

```
rag_based_study_assistant_v2/
├── study_assistant_v2.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── .env                  # Environment variables
└── README.md             # Project documentation
```

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- OpenAI for GPT-4 API
- LangChain community
- Streamlit team

## Contact 📧

Ravindu Pabasara Karunarathna - [karurpabe@gmail.com](mailto:karurpabe@gmail.com)
