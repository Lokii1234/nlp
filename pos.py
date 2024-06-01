import nltk
# Download NLTK resource for the universal tagset
nltk.download('universal_tagset')
# Rest of your code
from nltk import word_tokenize
text = "Natural Languiage Processing is a fascinating field of study."
tokens = word_tokenize(text)
tags = nltk.pos_tag(tokens, tagset="universal")
print(tags)
