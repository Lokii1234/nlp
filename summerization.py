from transformers import pipeline

def perform_summarization(text):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summary = summarizer(text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summary[0]['summary_text']

if __name__ == "__main__":
    input_text = """The warning you're seeing is related to some weights in the "dslim/bert-base-NER" model checkpoint not being used for the specific task """
    summarized_text = perform_summarization(input_text)
    print("Original Text:")
    print(input_text)
    print("\nSummarized Text:")
    print(summarized_text)
