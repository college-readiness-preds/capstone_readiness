#imports
from requests import get
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

def get_features(school_id):
    '''
    this function scrapes the TEA website for an individual school to get the specific data
    '''
    url1=f'https://rptsvr1.tea.texas.gov/cgi/sas/broker?_service=marykay&_program=perfrept.perfmast.sas&_debug=0&lev=C&id={school_id}&prgopt=reports%2Ftapr%2Fperformance.sas'
    url2=f'https://rptsvr1.tea.texas.gov/cgi/sas/broker?_service=marykay&_program=perfrept.perfmast.sas&_debug=0&ccyy=2022&lev=C&id={school_id}&prgopt=reports/tapr/student.sas'
    url3=f'https://rptsvr1.tea.texas.gov/cgi/sas/broker?_service=marykay&_program=perfrept.perfmast.sas&_debug=0&ccyy=2022&lev=C&id={school_id}&prgopt=reports/tapr/staff.sas'
    url4=f'https://rptsvr1.tea.texas.gov/cgi/sas/broker?_service=marykay&_service=appserv&_debug=0&_program=sfadhoc.new_Campus_actual21.sas&which_camp={school_id}'
    
    response1=get(url1)
    soup1 = BeautifulSoup(response1.content, 'html.parser')
    response2=get(url2)
    soup2 = BeautifulSoup(response2.content, 'html.parser')
    response3=get(url3)
    soup3 = BeautifulSoup(response3.content, 'html.parser')
    response4=get(url4)
    soup4 = BeautifulSoup(response4.content, 'html.parser')
    
    
    chinaspring={
        'school_id': [school_id],
        'eng1': [soup.find_all('td', class_='r t data')[2].text], 
        'eng2': [soup.find_all('td', class_='r t data')[98].text], 
        'algebra': [soup.find_all('td', class_='r t data')[194].text],
        'biology': [soup.find_all('td', class_='r t data')[290].text], 
        'history': [soup.find_all('td', class_='r t data')[386].text], 
        'ebel': [soup2.find_all('td', class_='r b data')[241].text],
        'econdis': [soup2.find_all('td', class_='r b data')[217].text],
        'salary': [soup3.find_all('td', class_='r b data')[-43].text],
        'masters': [soup3.find_all('td', class_='r b data')[-114].text],
        'doct': [soup3.find_all('td', class_='r b data')[-110].text],
        'beginning_teach': [soup3.find_all('td', class_='r b data')[-106].text],
        'ex_5': [soup3.find_all('td', class_='r b data')[-102].text],
        'ex_10': [soup3.find_all('td', class_='r b data')[-98].text],
        'ex_1120': [soup3.find_all('td', class_='r b data')[-94].text],
        'ex_2130': [soup3.find_all('td', class_='r b data')[-90].text],
        'ex_plus': [soup3.find_all('td', class_='r b data')[-86].text],
        'extra_fund': [soup4.find_all('td', class_='r data')[91].text],
        'all_fund': [soup4.find_all('td', class_='r data')[31].text],
        'ratio': [soup3.find_all('td', class_='r b data')[-83].text]
    }
    add=pd.DataFrame(chinaspring)
    return add


def all_schools(school_id):
    '''
    this function takes in school ids and concats each individual schools info into one dataframe
    '''
    all_info=pd.DataFrame({
        'school_id': [],
        'eng1': [], 
        'eng2': [], 
        'algebra': [],
        'biology': [], 
        'history': [], 
        'ebel': [],
        'econdis': [],
        'salary': [],
        'masters': [],
        'doct': [],
        'beginning_teach': [],
        'ex_5': [],
        'ex_10': [],
        'ex_1120': [],
        'ex_2130': [],
        'ex_plus': [],
        'extra_fund': [],
        'all_fund': [],
        'ratio': []
    })
    for n in school_id:
        try:
            school=get_features(n)
            all_info=pd.concat([all_info, school], ignore_index=True)
        except:
            continue
    return all_info