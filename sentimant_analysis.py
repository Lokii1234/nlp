from transformers import pipeline
def perform_sentiment_analysis(text):
    # Create a sentiment analysis pipeline
    sentiment_analysis = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
    # Perform sentiment analysis
    result = sentiment_analysis(text)
    # Return the result
    return result[0]['label'], result[0]['score']
if __name__ == "__main__":
    # Get user input dynamically
    input_text = input("Enter the text for sentiment analysis: ")
    # Perform sentiment analysis
    sentiment_label, sentiment_score = perform_sentiment_analysis(input_text)
    # Print the results
    print(f"Sentiment Label: {sentiment_label}")
    print(f"Sentiment Score: {sentiment_score}")
