import requests
from bs4 import BeautifulSoup
import time
from selenium import webdriver
import selenium
import pandas
import pickle
import os

chromedriver = "/Users/mehrdadshokrabadi/Downloads/chromedriver"

class web_scraping(object):
    def __init__(self, url_base, low_range_for_pages, high_range_for_pages):
        #Initialization of the data structure
        self.data = []
        self.Output = {}
        #Methods to fill in the data structure
        self.url_loopimg(url_base, low_range_for_pages, high_range_for_pages)

    def page_scraping(self, input_url):
        r = requests.get(input_url)
        soup = BeautifulSoup(r.content, 'html.parser')
        rows = soup.select('tbody tr')
        for row in rows:
            d = dict()
            d['NewsTitle'] = row.select_one('.views-field-name').text.strip()
            d['Field'] = row.select_one('.views-field-field-story-topic').text.strip()
            d['Date'] = row.select_one('.views-field-field-story-date').text.strip()
            self.data.append(d)

    def url_loopimg(self, url_base, low_range_for_pages ,high_range_for_pages ):
        url = []
        url.append(url_base)
        self.page_scraping(url[0])
        for i in range(low_range_for_pages,high_range_for_pages):
            print(i)
            File = url_base + '?page=' + str(i);
            url.append(File)
            self.page_scraping(url[i])
            time.sleep(1)
        self.Output['WebScrapingOutputs_NewsTitles'] = pandas.DataFrame(data=self.data);


class web_scraping_newsContent(object):
    def __init__(self, url_base, low_range_for_pages, high_range_for_pages):
        #Initialization of the data structure
        self.data = []
        self.Output = {}
        #Methods to fill in the data structure
        self.url_looping(url_base, low_range_for_pages, high_range_for_pages)

    def page_scraping(self, input_url):
        r = requests.get(input_url)
        soup = BeautifulSoup(r.content, 'html.parser')
        rows = soup.select('div[class="quicktabs-views-group"]')
        for row in rows:
            d = dict()
            d['NewsTitle'] = row.select_one('.news-title').text.strip()
            d['Source'] = row.select_one('.news-source').text.strip()
            d['Bias'] = row.select_one('.bias-image img')['title'].split(': ')[-1]
            d['news_body'] = row.select_one('.news-body').text.strip()
            self.data.append(d)

    def Open_and_Click_On_Webpage(self, input_url):
        #browser = webdriver.Safari()
        browser = webdriver.Chrome(chromedriver)
        browser.get(input_url)
        time.sleep(1)
        links = browser.find_elements_by_css_selector("a[href*='/story/']")
        Counter = 0
        while Counter < len(links):
            try:
                links[Counter].click()
                browser.switch_to_window(browser.window_handles[0])
                self.page_scraping(browser.current_url)
                time.sleep(1)
                browser.back();
                links = browser.find_elements_by_css_selector("a[href*='/story/']")
                Counter += 1
            except IndexError:
                print('IndexError')
                browser.get(input_url)
                links = browser.find_elements_by_css_selector("a[href*='/story/']")
                Counter += 1
                pass
            except selenium.common.exceptions.ElementClickInterceptedException:
                print('ElementClickInterceptedException')
                browser.get(input_url)
                links = browser.find_elements_by_css_selector("a[href*='/story/']")
                Counter += 1
                pass
            except selenium.common.exceptions.StaleElementReferenceException:
                print('StaleElementReferenceException')
                browser.get(input_url)
                links = browser.find_elements_by_css_selector("a[href*='/story/']")
                Counter += 1
                pass
        #browser.quit()

    def url_looping(self, url_base, low_range_for_pages ,high_range_for_pages ):
        url = []
        # url.append(url_base)
        # self.Open_and_Click_On_Webpage(url[0])
        for i in range(low_range_for_pages,high_range_for_pages):
            print(i)
            File = url_base + '?page=' + str(i);
            url.append(File)
            self.Open_and_Click_On_Webpage(url[i])
            time.sleep(1)
            # Saving the data structures
            BaseDirectory = '/Users/mehrdadshokrabadi/Documents/Miscellaneous/DS/Insight/Project'
            WebScrapingSaveDir = BaseDirectory + '/Web Scraping'
            WebScrapingOutputs_NewsBodies = self.data;
            os.chdir(WebScrapingSaveDir)
            f = open("WebScrapingOutputs_NewsBodies.pkl","wb")
            pickle.dump(WebScrapingOutputs_NewsBodies,f)
            f.close()
        self.Output['WebScrapingOutputs_NewsBodies'] = pandas.DataFrame(data=self.data);



