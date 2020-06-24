from __future__ import absolute_import, print_function
import os
import pickle
import pandas as pd
import WebScraping_Classes

############################################################
#                        Define Inputs                     #
############################################################
BaseDirectory = '/Users/mehrdadshokrabadi/Documents/Miscellaneous/DS/Insight/Project'
url_base_NewsTitles = 'https://www.allsides.com/story/admin'
url_base_NewsBodies = 'https://www.allsides.com/story/admin'
low_range_for_pages = 1
high_range_for_pages = 90


# #############################################################################
# #                        Saving Outputs of Web Scraping                     #
# #############################################################################
# WebScrapingSaveDir = BaseDirectory + '/Web Scraping'
# WebScraping = WebScraping_Classes.web_scraping(url_base_NewsTitles, low_range_for_pages, high_range_for_pages)
#
# # Saving the data structures
# WebScrapingOutputs_NewsTitles = WebScraping.Output['WebScrapingOutputs_NewsTitles']
# os.chdir(WebScrapingSaveDir)
# f = open("WebScrapingOutputs_NewsTitles.pkl","wb")
# pickle.dump(WebScrapingOutputs_NewsTitles,f)
# f.close()

#############################################################################
#                  Saving Outputs of Web Scraping - News Body               #
#############################################################################
WebScrapingSaveDir = BaseDirectory + '/Web Scraping'
WebScraping = WebScraping_Classes.web_scraping_newsContent(url_base_NewsBodies, low_range_for_pages, high_range_for_pages)

# Saving the data structures
WebScrapingOutputs_NewsBodies = WebScraping.Output['WebScrapingOutputs_NewsBodies']
os.chdir(WebScrapingSaveDir)
f = open("WebScrapingOutputs_NewsBodies.pkl","wb")
pickle.dump(WebScrapingOutputs_NewsBodies,f)
f.close()
