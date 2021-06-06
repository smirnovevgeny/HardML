import scrapy
from urllib.parse import urljoin

class ActorItem(scrapy.Item):
    bio = scrapy.Field()
    born = scrapy.Field()
    movies = scrapy.Field()
    name = scrapy.Field()
    url = scrapy.Field()

class MovieInfo(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    cast = scrapy.Field()

IMDB_LINK = "https://www.imdb.com"

class ActorSpider(scrapy.Spider):
    name = "actor_imdb"
    allowed_domains = ["imdb.com"]
    start_urls = ['https://www.imdb.com/search/name/?gender=male%2Cfemale&ref_=nv_cel_m']
   
    def parse(self, response):
        # получение таблицы со строками, хараактеризующими ТОП-фильмы        
        for raw in response.xpath('.//*[@class="lister-list"]/div'):
            url_suffix = raw.xpath('.//*[@class="lister-item-header"]/a/@href').extract_first()
            url = urljoin(IMDB_LINK, url_suffix)
            yield scrapy.Request(url, callback=self.parseActorInfo, meta={'url' : url})
    
    def parseActorInfo(self, response):
        actor_info = ActorItem()
        actor_info["bio"] = response.xpath('//*[@class="inline"]/text()').extract_first().strip()
        actor_info["born"] = response.xpath(".//*/div[@id='name-born-info']/time/@datetime").extract_first()
        actor_info["movies"] = list(map(lambda x: x.extract(), response.xpath('//*/b/a/text()')[:15]))
        movies = []
        for url_info in response.xpath('//*/b/a')[:15]:
            title = url_info.xpath("./text()").extract_first()
            movies.append(title)
            url = urljoin(IMDB_LINK, url_info.xpath("./@href").extract_first())
#             yield scrapy.Request(url, callback=self.parseCast, meta={"url": url, "title": title})
        actor_info["movies"] = movies
        actor_info["name"] = response.xpath('//*[@class="header"]/span[@class="itemprop"]/text()').extract_first()
        actor_info['url'] = response.meta['url'] 
        return actor_info
    
    def parseCast(self, response):
        movie_info = MovieInfo()
        movie_info["url"] = response.meta["url"]
        movie_info["title"] = response.meta["title"]
        cast = []
        for item in response.xpath('//*/a'):
            item_url = item.xpath("./@href").extract_first()
            if item_url and item_url.startswith("/name/nm") and "tt_cl_t" in item_url:
                cast.append(item.xpath("./text()").extract_first())
        movie_info['cast'] = cast
        return movie_info



from scrapy.crawler import CrawlerProcess
process = CrawlerProcess(
    settings={
        "FEEDS": {
            "items1.json": {"format": "json"},
        }
    }
)

process.crawl(ActorSpider)
process.start()

import json
info = json.load(open("items1.json", "r"))
with open('second_task.jl', 'w') as outfile:
    for entry in info:
        if "cast" in entry:
            json.dump(entry, outfile)
            outfile.write('\n')
