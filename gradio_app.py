import gradio as gr
from theme_classifier import themeClassifier
from character_network import named_entity_recog,characterNetwork
import os

def get_themes(theme_list,subtitles_path,save_path):
    theme_list = theme_list.split(',')
    classifier = themeClassifier(theme_list)
    op_df = classifier.get_themes(subtitles_path,save_path)

    #remove dialogue from theme list
    theme_list = [theme for theme in theme_list if theme.strip().lower() != 'dialogue']
    op_df = op_df[theme_list]

    op_df = op_df.sum().reset_index()
    op_df.columns = ['Theme','Score']

    op_chart = gr.BarPlot(op_df,x='Theme',y='Score',title='Series Theme Score',tooltip=['Theme','Score'],height=320)
    
    return op_chart

def get_char_network(subtitles_path,NER_save_path):
    NER_obj = named_entity_recog()
    NER_df = NER_obj.get_ners(dataset_path=subtitles_path,save_path=NER_save_path)
    charNetwork_obj = characterNetwork()
    relation_df = charNetwork_obj.generate_character_network(NER_df)
    html_embed = charNetwork_obj.draw_character_network(relation_df)

    return html_embed


def main():
    with gr.Blocks() as iface:
        #theme classification
        gr.HTML("<h1>Theme Classification (with Zero Shot Classifier)</h1>")
        with gr.Row():
            with gr.Column():
                plot = gr.BarPlot()
            with gr.Column():
                theme_list = gr.Textbox(label='Themes')
                subtitles_path = gr.Textbox(label='Subtitles Path')
                save_path = gr.Textbox(label='Save Path')
                submit_button = gr.Button("Get Themes")
                submit_button.click(get_themes,inputs=[theme_list,subtitles_path,save_path],outputs=[plot])

        #character network
        gr.HTML("<h1>Character Network</h1>")
        with gr.Row():
            with gr.Column():
                character_graph = gr.HTML()
            with gr.Column():
                subtitles_path = gr.Textbox(label='Subtitles Path')
                NER_save_path = gr.Textbox(label='Network Save Path')
                NER_submit_button = gr.Button("Generate Network")
                NER_submit_button.click(get_char_network,inputs=[subtitles_path,NER_save_path],outputs=[character_graph])

                
    iface.launch(share=True)

if __name__ == '__main__':
    main()