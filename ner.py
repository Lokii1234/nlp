from transformers import pipeline
def perform_ner_with_dostilcamembert(text):
    ner = pipeline(
        task='ner',
        model="cmarkea/distilcamembert-base-ner",
        tokenizer="cmarkea/distilcamembert-base-ner",
        aggregation_strategy ="simple"
    )
    result = ner(text)
    return result if result else []
def display_ner_results(ner_results):
    if not ner_results:
        return ["No entities found"]
    grouped_results = {}
    for entity in ner_results:
        entity_group = entity['entity_group']
        entity_word = entity['word']
        if entity_group not in grouped_results:
            grouped_results[entity_group] = []
        grouped_results[entity_group].append(entity_word)
    formatted_results = [f"{key}: {', '.join(value)}" for key, value in grouped_results.items()]
    return formatted_results
def map_entity_group(entity_group):
    return entity_group
