import torch
from sentence_transformers import util, SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


class llm:
  def __init__(self,embedding_model_used,llm_model_used, document_data, use_quantization=False):
    self.embedding_model_used=embedding_model_used
    self.embedding_model=SentenceTransformer(model_name_or_path=self.embedding_model_used,device=device)
    self.llm_model_used=llm_model_used
    self.use_q=use_quantization
    self.document_data=document_data
    self.token=AutoTokenizer.from_pretrained(llm_model_used)
    text_chunk=[item["sentence_chunk"] for item in document_data]
    self.embedding_data=self.embedding_model.encode(text_chunk,convert_to_tensor=True,show_progress_bar=True)
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    self.llm_model=AutoModelForCausalLM.from_pretrained(self.llm_model_used,torch_dtype=torch.float16, quantization_config= quantization_config if self.use_q else None, low_cpu_mem_usage=False).to(device)

  def retrieve_relevant_resources(self, query, embeddings, n_resources_to_return=5):

    query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
    return scores, indices

  def prompt_formatter(self, query, context_items):

    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
\nExample 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
\nExample 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    # Update base prompt with context items and query
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = self.token.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt

  def run(self, query, n_resources_to_return=5):
    scores, indices = self.retrieve_relevant_resources(query=query, embeddings= self.embedding_data, n_resources_to_return=n_resources_to_return)
    context_items = [self.document_data[i] for i in indices]
    prompt = self.prompt_formatter(query=query, context_items=context_items)
    input_ids = self.token(prompt, return_tensors="pt").to("cuda")

    outputs = self.llm_model.generate(**input_ids, do_sample=True, max_new_tokens=512)
    output_text = self.token.decode(outputs[0])
    return output_text

