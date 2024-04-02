import gradio as gr
from service import Service

def consult_bot(message, history):
    service = Service()
    return service.answer(message, history)

css = '''
/* Container adjustments for optimal width and centered alignment */
.gradio-container { 
    max-width: 900px !important; 
    margin: 40px auto !important; 
    border-radius: 8px; 
    box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
}

.user-message {
    background-color: #e0f7fa !important; /* Light blue background for user messages */
    color: #005662 !important; /* Darker text color for contrast */
}

/* Chatbot response styling */
.bot-message {
    background-color: #fce4ec !important; /* Light pink background for bot messages */
    color: #880e4f !important; /* Darker text color for contrast */
}

/* Chat input box styling */
input[type="text"] {
    border-radius: 20px !important;
    padding: 10px 20px;
}

/* Button styling for a more modern look */
button {
    border-radius: 20px !important;
    padding: 8px 20px;
    border: none;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.3s ease;
}

button:hover {
    opacity: 0.8;
}

/* Custom theme color adjustments */
:root {
    --primary-color: #4a56e2; /* Primary color for buttons and highlights */
    --text-color: #333; /* Default text color for better contrast */
    --bg-color: #fff; /* Background color for chat bubbles */
}
'''

demo = gr.ChatInterface(
    css=css,
    fn=consult_bot, 
    title='Consultation Chat Bot',
    chatbot=gr.Chatbot(height=500, bubble_full_width=False),  # Increase height for more chat visibility
    theme=gr.themes.Default(spacing_size='md', radius_size='md'),  # Adjust spacing and radius for a modern look
    textbox=gr.Textbox(placeholder="Type your question here...", container=False, scale=7),
    examples=[
        'Hi, can you introduce yourself?',
        'Who proposed this framework?',
        'What are the contact details of the team members?',
        'What is the biggest animal in the world?'
    ],
    submit_btn=gr.Button('Ask', variant='primary'),
    clear_btn=gr.Button('Start Over', variant='secondary'),
    # Opting for meaningful button labels for user clarity
    retry_btn=None,  # Consider if retry functionality is needed based on use case
    undo_btn=None,  # Same as above for undo functionality
)

if __name__ == '__main__':
    demo.launch(share=True)