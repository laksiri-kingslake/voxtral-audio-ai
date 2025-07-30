import gradio as gr
import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration
import spaces

#### Functions

@spaces.GPU
def process_transcript(language: str, audio_path: str) -> str:
    """Process the audio file to return its transcription.

    Args:
        language: The language of the audio.
        audio_path: The path to the audio file.

    Returns:
        The transcribed text of the audio.
    """

    if audio_path is None:
        return "Please provide some input audio: either upload an audio file or use the microphone."
    else:
        id_language = dict_languages[language]
        inputs = processor.apply_transcrition_request(language=id_language, audio=audio_path, model_id=model_name)
        inputs = inputs.to(device, dtype=torch.bfloat16)
        outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
        decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        return decoded_outputs[0]
###

@spaces.GPU
def process_translate(language: str, audio_path: str) -> str:
    if audio_path is None:
        return "Please provide some input audio: either upload an audio file or use the microphone."
    else:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path": audio_path,
                    },
                    {"type": "text", "text": "Translate this in "+language},
                ],
            }
        ]
        
        inputs = processor.apply_chat_template(conversation)
        inputs = inputs.to(device, dtype=torch.bfloat16)
        
        outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
        decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        return decoded_outputs[0]
###

@spaces.GPU
def process_chat(question: str, audio_path: str) -> str:
    if audio_path is None:
        return "Please provide some input audio: either upload an audio file or use the microphone."
    else:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path": audio_path,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        inputs = processor.apply_chat_template(conversation)
        inputs = inputs.to(device, dtype=torch.bfloat16)
        
        outputs = model.generate(**inputs, max_new_tokens=500)
        decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return decoded_outputs[0]
###

def disable_buttons():
    return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

def enable_buttons():
    return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
###

### Initializations

MAX_TOKENS = 32000

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"*** Device: {device}")
model_name = 'mistralai/Voxtral-Mini-3B-2507'

processor = AutoProcessor.from_pretrained(model_name)
model = VoxtralForConditionalGeneration.from_pretrained(model_name,
                                                        torch_dtype=torch.bfloat16,
                                                        device_map=device)
# Supported languages
dict_languages = {"English": "en",
                  "French": "fr",
                  "German": "de",
                  "Spanish": "es",
                  "Italian": "it",
                  "Portuguese": "pt",
                  "Dutch": "nl",
                  "Hindi": "hi"}


#### Gradio interface
with gr.Blocks(title="Voxtral") as voxtral:
    gr.Markdown("# **Voxtral Mini Evaluation**")
    gr.Markdown("""#### Voxtral Mini is an enhancement of **Ministral 3B**, incorporating state-of-the-art audio input \
    capabilities while retaining best-in-class text performance. 
    #### It excels at speech transcription, translation and audio understanding.""")
    
    with gr.Accordion("🔎 More on Voxtral", open=False):
        gr.Markdown("""## **Key Features:**

#### Voxtral builds upon Ministral-3B with powerful audio understanding capabilities.
##### - **Dedicated transcription mode**: Voxtral can operate in a pure speech transcription mode to maximize performance. By default, Voxtral automatically predicts the source audio language and transcribes the text accordingly
##### - **Long-form context**: With a 32k token context length, Voxtral handles audios up to 30 minutes for transcription, or 40 minutes for understanding
##### - **Built-in Q&A and summarization**: Supports asking questions directly through audio. Analyze audio and generate structured summaries without the need for separate ASR and language models
##### - **Natively multilingual**: Automatic language detection and state-of-the-art performance in the world’s most widely used languages (English, Spanish, French, Portuguese, Hindi, German, Dutch, Italian)
##### - **Function-calling straight from voice**: Enables direct triggering of backend functions, workflows, or API calls based on spoken user intents
##### - **Highly capable at text**: Retains the text understanding capabilities of its language model backbone, Ministral-3B""")

    
    gr.Markdown("### **1. Upload an audio file, record via microphone, or select a demo file:**")
    gr.Markdown("### *(Voxtral handles audios up to 30 minutes for transcription)*")

    with gr.Row():
        sel_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", 
                             label="Set an audio file to process it:")
        example = [["mapo_tofu.mp3"]]
        gr.Examples(
            examples=example,
            inputs=sel_audio,
            outputs=None,
            fn=None,
            cache_examples=False,
            run_on_click=False
        )

    with gr.Row():
        gr.Markdown("### **2. Choose one of theese tasks:**")
        
    with gr.Row():
        with gr.Column():
            with gr.Accordion("📝 Transcription", open=True):
                sel_language = gr.Dropdown(
                    choices=list(dict_languages.keys()),
                    value="English",
                    label="Select the language of the audio file:"
                )
                submit_transcript = gr.Button("Extract transcription", variant="primary")
                text_transcript = gr.Textbox(label="💬 Generated transcription", lines=10)

        with gr.Column():
            with gr.Accordion("🔁 Translation", open=True):
                sel_translate_language = gr.Dropdown(
                    choices=list(dict_languages.keys()),
                    value="English",
                    label="Select the language for translation:"
                )
    
                submit_translate = gr.Button("Translate audio file", variant="primary")
                text_translate = gr.Textbox(label="💬 Generated translation", lines=10)

        with gr.Column():
            with gr.Accordion("🤖 Ask audio file", open=True):
                question_chat = gr.Textbox(label="Enter your question about audio file:", placeholder="Enter your question about audio file")
                submit_chat = gr.Button("Ask audio file", variant="primary")
                example_chat = [["What is the subject of this audio file?"], ["Quels sont les ingrédients ?"]]
                gr.Examples(
                    examples=example_chat,
                    inputs=question_chat,
                    outputs=None,
                    fn=None,
                    cache_examples=False,
                    run_on_click=False
                )
                text_chat = gr.Textbox(label="💬 Model answer", lines=10)
            
### Processing
    
    # Transcription
    submit_transcript.click(
        disable_buttons,
        outputs=[submit_transcript, submit_translate, submit_chat],
        trigger_mode="once",
    ).then(
        fn=process_transcript,
        inputs=[sel_language, sel_audio],
        outputs=text_transcript
    ).then(
        enable_buttons,
        outputs=[submit_transcript, submit_translate, submit_chat],
    )

    # Translation
    submit_translate.click(
        disable_buttons,
        outputs=[submit_transcript, submit_translate, submit_chat],
        trigger_mode="once",
    ).then(
        fn=process_translate,
        inputs=[sel_translate_language, sel_audio],
        outputs=text_translate
    ).then(
        enable_buttons,
        outputs=[submit_transcript, submit_translate, submit_chat],
    )

    # Chat
    submit_chat.click(
        disable_buttons,
        outputs=[submit_transcript, submit_translate, submit_chat],
        trigger_mode="once",
    ).then(
        fn=process_chat,
        inputs=[question_chat, sel_audio],
        outputs=text_chat
    ).then(
        enable_buttons,
        outputs=[submit_transcript, submit_translate, submit_chat],
    )
    
### Launch the app

if __name__ == "__main__":
    voxtral.queue().launch()
