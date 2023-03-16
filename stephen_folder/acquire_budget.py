#imports
from requests import get
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

def get_features(school_id):
    '''
    this function scrapes the TEA website for an individual school to get the specific data
    '''
    
    url4=f'https://rptsvr1.tea.texas.gov/cgi/sas/broker?_service=marykay&_service=appserv&_debug=0&_program=sfadhoc.new_Campus_actual21.sas&which_camp={school_id}'
          
    
    response4=get(url4)
    soup4 = BeautifulSoup(response4.content, 'html.parser')
    
    
    chinaspring={
        #still need to change the index
        'Operating-Payroll': [school_id],
        'Other Operating': [soup4.find_all('td', class_='r data')[91].text],
        # for i in range () to get each elememt
        'Non-Operating(Equipt/Supplies)': [soup4.find_all('td', class_='r data')[31].text],
        'Instruction ': [soup4.find_all('td', class_='r data')[31].text],
        'Instructional Res/Media': [soup4.find_all('td', class_='r data')[31].text],
        'Curriculum/Staff Develop': [soup4.find_all('td', class_='r data')[31].text],
        'Instructional Leadership': [soup4.find_all('td', class_='r data')[31].text],
        'School Leadership': [soup4.find_all('td', class_='r data')[31].text],
        'Guidance/Counseling Svcs': [soup4.find_all('td', class_='r data')[31].text],
        'Social Work Services': [soup4.find_all('td', class_='r data')[31].text],
        'Health Services': [soup4.find_all('td', class_='r data')[31].text],
        'Food': [soup4.find_all('td', class_='r data')[31].text],
        'Extracurricular': [soup4.find_all('td', class_='r data')[31].text],
        'Plant Maint/Operation': [soup4.find_all('td', class_='r data')[31].text],
        'Security/Monitoring': [soup4.find_all('td', class_='r data')[31].text],
        'Data Processing Svcs': [soup4.find_all('td', class_='r data')[31].text],
        'Regular': [soup4.find_all('td', class_='r data')[31].text],
        'Gifted & Talented': [soup4.find_all('td', class_='r data')[31].text],
        'Career & Technical': [soup4.find_all('td', class_='r data')[31].text],
        'Students with Disabilities': [soup4.find_all('td', class_='r data')[31].text],
        'Accelerated Education': [soup4.find_all('td', class_='r data')[31].text],
        'Bilingual': [soup4.find_all('td', class_='r data')[31].text],
        'Nondisc Alted-AEP Basic Serv': [soup4.find_all('td', class_='r data')[31].text],
        'Disc Alted-DAEP Basic Serv': [soup4.find_all('td', class_='r data')[31].text],
        'Disc Alted-DAEP Supplemental': [soup4.find_all('td', class_='r data')[31].text],
        'T1 A Schoolwide-St Comp >=40%': [soup4.find_all('td', class_='r data')[31].text],
        'Athletic Programming': [soup4.find_all('td', class_='r data')[31].text],
        'High School Allotment': [soup4.find_all('td', class_='r data')[31].text],
        'Prekindergarten': [soup4.find_all('td', class_='r data')[31].text],
        'Early Education Allotment': [soup4.find_all('td', class_='r data')[31].text],
        'CCMR': [soup4.find_all('td', class_='r data')[31].text],


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