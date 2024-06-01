from data_prepocess import pdf_preprocess
from llm import llm

pdf_path="human-nutrition-text.pdf"
data_preprocess=pdf_preprocess(pdf_path)
text=data_preprocess.process()

embedding_model_used="all-mpnet-base-v2"
llm_model_used="google/gemma-7b-it"

llm=llm(embedding_model_used="all-mpnet-base-v2", llm_model_used="google/gemma-2b-it",document_data=text)

query="What are the macronutrients, and what roles do they play in the human body?"

output=llm.run(query)

print(output)