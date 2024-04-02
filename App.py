import gradio as gr
from service import Service

def consult_bot(message, history):
    service = Service()
    return service.answer(message, history)

css = '''
.gradio-container { max-width:850px !important; margin:20px auto !important;}
.message { padding: 10px !important; font-size: 14px !important;}
'''

demo = gr.ChatInterface(
    css = css,
    fn = consult_bot, 
    title = 'Consultation Chat Bot',
    chatbot = gr.Chatbot(height=400, bubble_full_width=False),
    theme = gr.themes.Default(spacing_size='sm', radius_size='sm'),
    textbox=gr.Textbox(placeholder="Input your query here", container=False, scale=7),
    examples = ['Hi, how are you?', 'Who proposed this framework?', 'What are the contact details of the team members?', 'What is the biggest animial in the world?'],
    submit_btn = gr.Button('Submit', variant='primary'),
    clear_btn = gr.Button('Clear'),
    retry_btn = None,
    undo_btn = None,
)

if __name__ == '__main__':
    demo.launch(share=True)