from transformers import GPT2LMHeadModel, GPT2Tokenizer
def generate_text_func(prompt, model_name="gpt2", max_length = 1000, temperature=1.0):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length = max_length,
        temperature = temperature,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k = 50,
        top_p = 0.95,
        early_stopping = True,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
if __name__ == "_main_":
    # Example prompt
    prompt = "Once upon a time"
    # Generate text
    generated_text = generate_text_func(prompt, model_name="gpt2", max_length=100, temperature=0.8)
    # Print the generated text
    print("Generated Text:")
    print(generated_text)
