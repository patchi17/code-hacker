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
#yield scrapy.Request(url=url, callback=self.parse_dir_contents)
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

#start_urls = [
#'http://dmoztools.net/Recreation/Drugs/Entheogens/',
#'http://dmoztools.net/Recreation/Drugs/Psychedelics/',
#'http://dmoztools.net/Recreation/Drugs/Cannabis/',
#'http://dmoztools.net/Recreation/Drugs/',
#'http://dmoztools.net/Recreation/Nudism/',
#'http://dmoztools.net/Recreation/Guns/Airguns/',
#'http://dmoztools.net/Recreation/Guns/Shooting/',
#'http://dmoztools.net/Recreation/Drugs/Cannabis/Cultivation/'

#]

#npages = 2

# This mimics getting the pages using the next button.
#for i in range(2, npages + 2):
# start_urls.append('http://dmoztools.net/Recreation/='+str(i)+'')

#def parse(self, response):
#for href in response.request.url
# add the scheme, eg http://
#url  = "https:" + href.extract()
#yield scrapy.Request(url, callback=self.parse_dir_contents)
