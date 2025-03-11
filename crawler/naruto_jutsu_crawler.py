import scrapy
from bs4 import BeautifulSoup

class BlogSpider(scrapy.Spider):    #Spider is a crawler that crawls multiple webpages
    name = 'narutoSpider'
    start_urls = ['https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu?limit=250&offset=0&_cat=Jutsu']

    def parse(self, response):
        for href in response.css('.smw-columnlist-container')[0].css("a::attr(href)").extract():   #iterating over contents in a page
            jutsu_page_data = scrapy.Request("https://naruto.fandom.com"+href,
                                                callback=self.parse_jutsu_page)
            yield jutsu_page_data

        for next_page in response.css('a.mw-nextlink'):    #iterating over pages
            yield response.follow(next_page, self.parse)
    
    def parse_jutsu_page(self,response):
        jutsu_name = response.css("span.mw-page-title-main::text").extract()[0]
        jutsu_name = jutsu_name.strip()

        div_selector = response.css("div.mw-parser-output")[0]
        div_html = div_selector.extract()

        soup = BeautifulSoup(div_html).find('div')

        jutsu_type = ""
        if soup.find('aside'):
            aside = soup.find('aside')

            for row in aside.find_all('div',{'class':'pi-data'}):
                if row.find('h3'):
                    if row.find('h3').text.strip() == 'Classification':
                        jutsu_type = row.find('div').text.strip()

        soup.find('aside').decompose()  #decompose to remove it
        jutsu_descp = soup.text.strip()
        jutsu_descp = jutsu_descp.split('Trivia')[0].strip()

        return {'jutsu_name':jutsu_name,'jutsu_type':jutsu_type,'jutsu_descp':jutsu_descp}


#run with command: scrapy runspider crawler_file_path -o output_save_path(.jsonl extension)
#output saved in JSONl file where each line a JSON object

