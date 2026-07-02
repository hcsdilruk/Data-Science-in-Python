from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:

    def load_pdf(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents