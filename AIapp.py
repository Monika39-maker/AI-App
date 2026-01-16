# from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

# The folder only contains PDF files — use a PDF glob so files are found.
# dir_loader = DirectoryLoader(
#     "C:/Users/mdangol/Downloads/InsuranceApp/AI-App/AIdata",
#     glob="**/*.pdf",
#     loader_cls=PyPDFLoader,
#     loader_kwargs={'encoding': 'utf-8'},
#     show_progress=True
# )
loader = PyPDFLoader("C:/Users/mdangol/Downloads/InsuranceApp/AI-App/AIdata/file.pdf")
documents = loader.load()
print(documents)
# print(documents[:1])  # uncomment to inspect the first document





