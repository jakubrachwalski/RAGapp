import nltk
nltk.download('punkt')

print(nltk.data.path)

from langchain_community.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader("../pdf-documents/John_F_Kennedy.pdf")
docs = loader.load()
print(docs[0].page_content)
