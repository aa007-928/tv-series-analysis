import pandas as pd
import networkx as nx
from pyvis.network import Network

class characterNetwork:
    def __init__(self,window_size=10,top_k=200):
        self.window_size = window_size
        self.top_k=top_k

    def generate_character_network(self,df):
        window_size = self.window_size
        ER_list = []
        for row in df['ners']:
            entities_in_window = []
            for sentence in row:
                entities_in_window.append(list(sentence))
                entities_in_window = entities_in_window[-window_size:]    #take those in window
                #flatten list 2D -> 1D
                entities_in_window_flat = sum(entities_in_window,[])

                for entity in sentence:
                    for window_entity in entities_in_window_flat:
                        if entity != window_entity:
                            ER_list.append(sorted([entity,window_entity]))    #sort as relation is not directed (same both ways)

        relation_df = pd.DataFrame({'value':ER_list})
        relation_df['source'] = relation_df['value'].apply(lambda x: x[0])
        relation_df['target'] = relation_df['value'].apply(lambda x: x[1])
        relation_df = relation_df.groupby(['source','target']).count().reset_index()
        relation_df = relation_df.sort_values('value',ascending=False)

        return relation_df
    
    def draw_character_network(self,relation_df):
        relation_df = relation_df.sort_values('value',ascending=False)
        relation_df = relation_df.head(self.top_k)

        G = nx.from_pandas_edgelist(relation_df,source='source',target='target',edge_attr='value',create_using=nx.Graph())

        net = Network(notebook=True,width='1000px',height='700px',bgcolor='#222222',font_color='white',cdn_resources='remote')
        node_degree = dict(G.degree)  

        nx.set_node_attributes(G,node_degree,'size')
        nx.set_node_attributes(G,'#FF7F00','color')
        net.from_nx(G)

        html = net.generate_html()
        html = html.replace("'","\"")
        html_embed = f"""<iframe style="width:100%; height:600px; margin:0 auto" name="NER_result" frameborder="0" srcdoc='{html}'></iframe>"""

        return html_embed