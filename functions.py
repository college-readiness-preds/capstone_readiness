#this function takes in a list of dollar values in strings and converts them to float

def convert_dollars_to_float(x):
    fix = []
    blah = []
    something = [] #this list holds only the balance keys from the dictionary
    only_dollars = [] # this holds the values without the commas
    convert = []
    for d in x:
        d = d.replace(' ', '')
        fix.append(d)
    for e in fix:
        e = e.replace('-', '')
        blah.append(e)
        # this holds the converted values 
    for a in blah:
        a = a.strip('$')
        something.append(a)
    for b in something:
        b = b.replace(",","")
        only_dollars.append(b)
    for c in only_dollars:
        c = float(c)
        convert.append(c)

    return convert

def change_dollars(df):
    df = df[df['salary'] != '-']
    df = df[df['salary'] != '?']
    df.salary = convert_dollars_to_float(df.salary)
    df.all_fund = convert_dollars_to_float(df.all_fund)
    df.extra_fund = convert_dollars_to_float(df.extra_fund)
    return df