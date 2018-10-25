import scrapy
from recreation.items import RecreationItem
from datetime import datetime
import re
import csv

class RecreationSpider(scrapy.Spider):
    name = "dmoz"
    allowed_domains = ["dmoztools.net"]
    
    #def parse(self, response):
    with open("science.txt", "rt") as f:
        start_urls = [url.strip() for url in f.readlines()]
        f.close()

    def parse(self, response):
        sites = response.css('#site-list-content > div.site-item > div.title-and-desc')
        items = []
        
        for site in sites:
            item = RecreationItem()
            item['Title'] = site.css('a > div.site-title::text').extract_first().strip()
            item['Sites'] = site.xpath('a/@href').extract_first().strip()
            item['Description'] = site.css('div.site-descr::text').extract_first().strip()
            item['URL'] = response.request.url
            item['MCat'] = " ".join(site.xpath("//div[contains(@class, 'current-cat science')]//a[contains(@class, 'breadcrumb')]//text()").extract()).split(' ', 2)[1]
            item['SCat'] = " ".join(site.xpath("//div[contains(@class, 'current-cat science')]//a[contains(@class, 'breadcrumb')]//text()").extract()).split(' ', 2)[2]
            item['CCat'] = " ".join(site.xpath("//b//text()").extract()).split(' ... ', 1)[1]
            items.append(item)
    
        return items
#item = RecreationItem()
#item['MCat'] = " ".join(response.xpath("//div[contains(@class, 'current-cat recreation')]//a[contains(@class, 'breadcrumb')]//text()").extract()).split(' ', 2)[1]
#item['SCat'] = " ".join(response.xpath("//div[contains(@class, 'current-cat recreation')]//a[contains(@class, 'breadcrumb')]//text()").extract()).split(' ', 2)[2]
#item['CCat'] = " ".join(response.xpath("//b//text()").extract()).split(' ... ', 1)[1]
#item['URL'] = response.request.url
#item['Title'] = "\n".join(response.xpath("//div[contains(@class, 'title-and-desc')]//a[contains(@target, '_blank')]//div[contains(@class, 'site-title')]//text()").extract())
#item['Sites'] = "\n".join(response.xpath("//div[contains(@class, 'title-and-desc')]/a[contains(@target, '_blank')]//@href").extract())
#item['Description'] = "\n".join(response.xpath("//div[contains(@class, 'title-and-desc')]/div[contains(@class, 'site-descr')]//text()").extract()).strip()        
#yield item
