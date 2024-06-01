from googletrans import Translator
from googletrans import LANGUAGES

def translate_text(text, target_language='en'):
    translator_instance = Translator()  # Change the variable name to avoid conflict
    translated_text = translator_instance.translate(text, dest=target_language)
    return translated_text.text
def get_language_options():
    
    language_options = [{'label': lang_name, 'value': lang_code} for lang_code, lang_name in LANGUAGES.items()]
    return language_options

