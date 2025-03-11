import gradio as gr
from theme_classifier import themeClassifier
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

    op_chart = gr.BarPlot(op_df,x='Theme',y='Score',title='Series Theme Score',tooltip=['Theme','Score'],height=330)
    
    return op_chart

def main():
    with gr.Blocks() as iface:
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
    iface.launch(share=True)

if __name__ == '__main__':
    main()