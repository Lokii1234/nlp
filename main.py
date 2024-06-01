import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from transformers import pipeline
from sentiment_analysis import perform_sentiment_analysis
from ner import perform_ner_with_dostilcamembert, display_ner_results
from summarization import perform_summarization
from translation import translate_text, get_language_options
from generation import generate_text_func
from image_generation import generate_and_show_image  
from PIL import Image
from io import BytesIO
import nltk
from nltk import word_tokenize

#Downloading NLTK data
nltk.download('universal_tagset')

#Initializing sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

#Initializing Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

#Function for part-of-speech tagging
def pos_tagging(text):
    tokens = word_tokenize(text)
    tags = nltk.pos_tag(tokens, tagset="universal")
    return tags

#Defining layout of the web application
app.layout = html.Div(style={'backgroundColor': '#f4f4f4', 'padding': '20px', 'height': '100vh'}, children=[
    html.H1("NLP Analysis Dashboard", style={'color': '#333', 'textAlign': 'center', 'textShadow': '2px 2px 5px rgba(0, 0, 0, 0.3)'}),
    html.Div(children=[
        html.H2("Task Selection", style={'color': '#333', 'textAlign': 'center', 'textShadow': '2px 2px 5px rgba(0, 0, 0, 0.3)'}),
        dcc.Textarea(
            id='textarea-input',
            placeholder='Enter text for analysis...',
            value='',
            style={'width': '100%', 'height': 100, 'marginBottom': '20px', 'border': '1px solid #ccc', 'borderRadius': '8px', 'padding': '8px', 'boxShadow': '2px 2px 5px rgba(0, 0, 0, 0.1)'}
        ),
        dcc.Dropdown(
            id='dropdown-input',
            options=[
                {'label': 'Sentiment Analysis', 'value': 'sentiment'},
                {'label': 'NER Analysis', 'value': 'ner'},
                {'label': 'Summarization', 'value': 'summarization'},
                {'label': 'Translation', 'value': 'translation'},
                {'label': 'Text Generation', 'value': 'generation'},
                {'label': 'Part of Speech Analysis', 'value': 'pos'},
                {'label': 'Generate Image', 'value': 'generate_image'}
            ],
            value='sentiment',
            style={'width': '50%', 'marginBottom': '20px'},
        ),
        html.Button('Submit', id='submit-button', n_clicks=0,
                    style={'marginBottom': '20px', 'border': 'none', 'borderRadius': '8px', 'padding': '10px 20px', 'background': '#4CAF50', 'color': 'white', 'cursor': 'pointer', 'boxShadow': '2px 2px 5px rgba(0, 0, 0, 0.3)'}),
        dbc.Button('Translate', id='open-modal-button', className='ml-auto', style={'display': 'none'}),
        dcc.Loading(
            id="loading-generate-image",
            type="default",
            children=[
                html.Div(id='output-container', style={'fontSize': '18px', 'fontWeight': 'bold', 'textShadow': '2px 2px 5px rgba(0, 0, 0, 0.3)', 'textAlign': 'center', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)', 'background': '#fff'}),
                html.Div(id='image-output', style={'textAlign':'center'})
            ]        
        ),
        # Modal for language selection
        dbc.Modal(
            id='language-modal',
            children=[
                dbc.ModalHeader('Select Target Language'),
                dbc.ModalBody([
                    dcc.Dropdown(
                        id='language-dropdown',
                        options=get_language_options(),
                        multi=False,
                        style={'width': '100%'}
                    ),
                ]),
                dbc.ModalFooter([
                    dbc.Button('Translate', id='translate-button', className='ml-auto'),
                    dbc.Button('Close', id='close-modal-button', className='mr-auto')
                ]),
            ],
        ),

        html.Div(id='hidden-translation-result', style={'display': 'none'}),
    ]),
])

# Callback to toggle visibility of translate button
@app.callback(
    Output('open-modal-button', 'style'),
    [Input('dropdown-input', 'value')]
)
def toggle_translate_button(selected_task):
    if selected_task == 'translation':
        return {'display': 'block'}
    else:
        return {'display': 'none'}

# Callback to toggle visibility of language modal
@app.callback(
    Output('language-modal', 'is_open'),
    [Input('open-modal-button', 'n_clicks'),
     Input('close-modal-button', 'n_clicks')],
    [State('language-modal', 'is_open')]
)
def toggle_modal(open_modal_clicks, close_modal_clicks, is_modal_open):
    ctx = dash.callback_context
    if ctx.triggered_id == 'open-modal-button':
        return True
    elif ctx.triggered_id == 'close-modal-button':
        return False
    return is_modal_open

# Callback to perform translation on modal button click
@app.callback(
    Output('hidden-translation-result', 'children'),
    [Input('translate-button', 'n_clicks')],
    [State('textarea-input', 'value'),
     State('language-dropdown', 'value')]
)
def translate_on_modal(n_clicks, textarea_value, target_lang):
    if n_clicks is not None and n_clicks > 0:
        if not target_lang:
            raise PreventUpdate
        translated_text = translate_text(textarea_value, target_language=target_lang)
        return translated_text
    return None

# Callback to update output based on selected task and input text
@app.callback(
    [Output('output-container', 'children'),
     Output('image-output', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('textarea-input', 'value'),
     State('dropdown-input', 'value'),
     State('hidden-translation-result', 'children')]
)
def update_output(n_clicks, textarea_value, selected_task, translation_result):
    if n_clicks > 0:
        if selected_task == 'sentiment':
            return perform_sentiment_analysis(textarea_value), None
        elif selected_task == 'ner':
            return display_ner_results(perform_ner_with_dostilcamembert(textarea_value)), None
        elif selected_task == 'summarization':
            return f'Summarized Text: {perform_summarization(textarea_value)}', None
        elif selected_task == 'translation':
            return f'Translated Text: {translation_result}', None
        elif selected_task == 'generation':
            generated_text = generate_text_func(prompt=textarea_value, model_name="gpt2", max_length=200, temperature=0.8)
            return html.P(f'Generated Text: {generated_text}'), None
        elif selected_task == 'pos':
            pos_result = pos_tagging(textarea_value)
            pos_formatted = ', '.join([f"{word} : {pos}" for word, pos in pos_result])
            return html.P(f'Part of Speech Tags: {pos_formatted}'), None
        elif selected_task == 'generate_image':
            img_path = generate_and_show_image(textarea_value)
            if img_path:
                img = html.Img(src=img_path, style={'max-width': '100%'})
                return None, img
            else:
                return None, html.P("Image generated.")
    return None, None

# Main execution block
if __name__ == '__main__':
    app.run_server(debug=True)
