#imports
from requests import get
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd


##############################################
#...........For Budget Data Only.............#
##############################################


def get_budgets():

    '''
    so far this only works for one url.  modify later to pass in a list of urls
    '''

    url = 'https://rptsvr1.tea.texas.gov/cgi/sas/broker?_service=marykay&_service=appserv&_debug=0&_program=sfadhoc.new_Campus_actual21.sas&which_camp=161920001'

    data = pd.read_html(url)
    data = data[1]
    

    return data


    

def get_school_ids():
    ''' Reads csv to get ids and sends to list so that it can be passed into acquire function'''
    high_school_ids = pd.read_csv('high_school_ids_clean.csv')
    high_schol_ids = high_school_ids.school_id.tolist()

    return high_schol_ids


