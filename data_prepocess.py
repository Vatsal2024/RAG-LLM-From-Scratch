import pandas as pd
import os
import requests
import fitz
from tqdm.auto import tqdm
import re
from spacy.lang.en import English

class pdf_preprocess:
  def __init__(self,pdf_path,num_sentence_chunk_size=10):
    self.pdf_path=pdf_path
    self.num_sentence_chunk_size=num_sentence_chunk_size

  def split_sent(self,input_list, slice_size):
    return [input_list[i:i+slice_size] for i in range(0,len(input_list),slice_size)]

  def text_formatter(self,text):
    return text.replace('\n',' ').strip()
  def reading_pdf(self):
    doc=fitz.open(self.pdf_path)
    pages_and_texts=[]
    for page_number, page in tqdm(enumerate(doc)):
        text=page.get_text()
        text=self.text_formatter(text)
        pages_and_texts.append({
            "page_number": page_number-41,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count": len(text.split(". ")),
            "page_token_count": len(text)/4,
            "text": text
        })


    nlp=English()
    nlp.add_pipe('sentencizer')

    for item in tqdm(pages_and_texts):
        item["sentences"]=list(nlp(item["text"]).sents)
        item["sentences"]=[str(sentence) for sentence in item["sentences"]]

        item["page_sentence_count_spacy"]=len(item["sentences"])

    return pages_and_texts


  def process(self,min_token_length=30):
    data=self.reading_pdf()
    for item in tqdm(data):
      item["sentence_chunks"]=self.split_sent(item['sentences'], slice_size=self.num_sentence_chunk_size)
      item["num_chunks"]=len(item["sentence_chunks"])

    pages_and_chunks = []
    for item in tqdm(data):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]

            # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get stats about the chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters

            pages_and_chunks.append(chunk_dict)

    df = pd.DataFrame(pages_and_chunks)
    pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")

    return pages_and_chunks_over_min_token_len
