from bs4 import BeautifulSoup

class Cleaner:
    def __init__(self):
        pass

    def lineBreak(self,text):
        return text.replace("</p>","</p>/n")
    
    def removeHTMLtags(self,text):
        return BeautifulSoup(text,"lxml").text

    def clean(self,text):
        text = self.lineBreak(text)
        text = self.removeHTMLtags(text)
        text = text.strip()
        return text