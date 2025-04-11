from transformers import pipeline, AutoModelForCausalLM

#model = AutoModelForCausalLM.from_pretrained("finetuned-llama")
pipeline = pipeline(task="text-generation", model="finetuned_llama", device=0)
output = pipeline("the secret to baking a really good cake is ")
print(output)
