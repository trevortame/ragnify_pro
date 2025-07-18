RAGNIFY - PDF Chatbot with LLaMA & ML Model Insights 🚀

Ragnify is a PDF-powered chatbot built with a Formula 1-inspired UI. It uses LLaMA LLM, LangChain, FAISS, and HuggingFace Embeddings to perform Q&A over uploaded PDF documents. It also offers ML model-based insights to identify the most relevant answers using Linear Regression, Random Forest, and XGBoost.

🔥 Features

📤 Upload and parse multiple PDFs (with OCR fallback)

🧠 Ask questions and get answers from LLaMA 2

📑 Highlight keywords across selected PDFs

💡 Visual PDF display with keyword highlights

📊 ML model scoring with multiple regression models

💬 Downloadable chat history

⚙️ Tech Stack

Frontend: Streamlit

LLM: LLaMA 2 (using GGUF format via llama-cpp-python)

Embeddings: HuggingFace (all-MiniLM-L6-v2)

Vector DB: FAISS

OCR: PyMuPDF + Tesseract

ML Models: Scikit-learn, XGBoost

PDF Parsing: PyMuPDF, pdf2image, pytesseract

📦 Installation

1. Clone the Repository

git clone https://github.com/trevortame/ragnify_pro.git
cd ragnify_pro

2. Create Virtual Environment

python -m venv venv
venv\Scripts\activate  # For Windows

3. Install Dependencies

pip install -r requirements.txt

4. Set up Tesseract (for OCR)

Download Tesseract OCR

Update the path in main.py:

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

5. Run the App

streamlit run main.py

📂 Folder Structure

RAGNIFY/
├── models/                    # LLaMA GGUF model
├── stored_pdfs/              # Uploaded PDFs
├── chat_history/             # Chat logs
├── main.py                   # Main Streamlit App
├── requirements.txt          # Python dependencies

📝 To-Do / Future Features

📁 Support for TXT/DOCX

📈 Model accuracy charts

🔐 Authentication support

🌐 HuggingFace Hub model integration

🙌 Acknowledgements

LLaMA.cpp

LangChain

FAISS

Streamlit Lottie

📬 Contact

Project by: @trevortameFeel free to raise issues or suggestions!

Give this repo a ⭐ if you like it!

