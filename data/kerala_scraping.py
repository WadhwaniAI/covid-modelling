import tabula
import logging
import pandas as pd
import numpy as np
logging.getLogger('PDFBox').setLevel(logging.WARNING)


links = [
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/04/Daily-Bulletin-HFWD-English-April-1.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/Daily-Bulletin-HFWD-English-March-31.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/Daily-Bulletin-HFWD-English-30th-March.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/Daily-Bulletin-English-29th-March.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/Daily-Bulletin-HFWD-English-28th-March.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/Daily-Bulletin-English-27th-March-2.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bulm_26032020e.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_25032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_24032020-1.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_24032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_22032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_21032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_20032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_19032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_18032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_17032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_16032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/Daily-Bulletin-HFWD-English-March-15th-1.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/Daily-Bulletin-HFWD-English-March-14th.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/Daily-Bulletin-HFWD-English-March-13th.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/Daily-Bulletin-HFWD-English-March-12th.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_11032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_10032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_09032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_08032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_07032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_06032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_05032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_04032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_03032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_02032020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_01032020-1.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_29022020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_28022020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_27022020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_26022020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_25022020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_24022020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_23022020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_22022020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_21022020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_20022020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_19022020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_18022020.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_17022020-1.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_16022020-1.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_15022020-1.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_14022020-1.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_13022020-1.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_12022020-1.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_11022020-1.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_10022020-1.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_09022020-1.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_08022020-1.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_07022020-1.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_06022020_eng.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_05022020_eng.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_04022020_eng.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_03022020_eng.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_02022020_eng.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_01022020_eng.pdf",
	"http://dhs.kerala.gov.in/wp-content/uploads/2020/03/bule_31012020_eng.pdf",
]

dates = [
	"01042020",
	"31032020",
	"30032020",
	"29032020",
	"28032020",
	"27032020",
	"26032020",
	"25032020",
	"24032020",
	"23032020",
	"22032020",
	"21032020",
	"20032020",
	"19032020",
	"18032020",
	"17032020",
	"16032020",
	"15032020",
	"14032020",
	"13032020",
	"12032020",
	"11032020",
	"10032020",
	"09032020",
	"08032020", # 24
	"07032020",
	"06032020",
	"05032020",
	"04032020",
	"03032020",
	"02032020",
	"01032020",
	"29022020",
	"28022020",
	"27022020",
	"26022020",
	"25022020",
	"24022020",
	"23022020",
	"22022020",
	"21022020",
	"20022020",
	"19022020",
	"18022020",
	"17022020",
	"16022020",
	"15022020",
	"14022020",
	"13022020",
	"12022020",
	"11022020", # 50
	"10022020",
	"09022020",
	"08022020",
	"07022020",
	"06022020",
	"05022020",
	"04022020",
	"03022020",
	"02022020",
	"01022020",
	"31012020",
]

def load(i):
	t = tabula.read_pdf(links[i], pages = "all", multiple_tables = True)
	# t = [f for f in t if f.shape[0] > 8 and f.shape[1] >= 2]
	return t

def savecases(table, i):
	table.to_csv("kerala/cases/cases_{}.csv".format(dates[i]), index=False)

def saveinfo(table, i):
	table.to_csv("kerala/info/info_{}.csv".format(dates[i]), index=False)
	
# def saveinfo(table, i):
# 	table.to_csv("tn/info_{}.csv".format(i), index=False)

def droprows(df, n):
	df.drop(df.head(n).index, inplace=True)

def headerinfo(df):
	# df.columns = ['District', 'Under Observation', 'Home Isolation', 'Hospitalized, Symptomatic', 'Hospitalized']
	df.columns = ['District', 'Under Observation', 'Home Isolation', 'Hospitalized', 'Discharged from Home Isolation']

def header(df):
	try:
		df.columns = ['District', 'Positive Cases Admitted', 'Other Districts']
	except:
		df["n"] = pd.NaT
		df.columns = ['District', 'Positive Cases Admitted', 'Other Districts']


def dropna(df):
	df.dropna(axis = 1, how = 'all', inplace=True)

def droptail(df, n=1):
	df.drop(df.tail(n).index, inplace=True) 

def check_headers():
	links_dates = tuple(zip(links, dates))
	review = []
	import webbrowser
	for i, (l, date) in enumerate(links_dates):
		# webbrowser.open(l)
		table = pd.read_csv('kerala/info/info_{}.csv'.format(dates[i]))
		# print()
		# print("1: ['District', 'Under Observation', 'Home Isolation', 'Hospitalized', 'Hospitalized Today']")
		# print("2: ['District', 'Under Observation', 'Home Isolation', 'Hospitalized', 'Discharged from Home Isolation Today']")
		# print("3: something else")
		# print()
		# print (l, date)
		# print(table[:1])
		# var = int(input("which set of headers does this have?: "))
		if i < 24 or i >= 50:
			table.columns = ['District', 'Under Observation', 'Home Isolation', 'Hospitalized', 'Hospitalized Today']
			saveinfo(table, i)
		elif i >= 24 and i < 50:
			table.columns = ['District', 'Under Observation', 'Home Isolation', 'Hospitalized', 'Discharged from Home Isolation Today']
			saveinfo(table, i)
		else:
			table.columns = table.columns
			saveinfo(table, i)
			review.append((i,date,l, "out of range"))

	print (review)
	return review

def verify_totals():
	links_dates = tuple(zip(links, dates))
	review = []
	trues = np.full((15,), True)
	for i, (l, date) in enumerate(links_dates):
		table = pd.read_csv('kerala/new/info_{}.csv'.format(dates[i]))
		for c in table.columns[1:]:
			if not (table[-1:][c] == np.sum(table[:-1])[c])[14]:
				print ("inconsistency! {} {}".format(c, date))
				print ("{} - {}".format(table[-1:][c], np.sum(table[:-1])[c]))
				review.append((i,date,l, "inconsistency"))
	print (review)
	return review

def check_totals_to_source():
	import webbrowser
	links_dates = tuple(zip(links, dates))
	review = []
	trues = np.full((15,), True)
	for i, (l, date) in enumerate(links_dates):
		webbrowser.open(l)
		table = pd.read_csv('kerala/new/info_{}.csv'.format(dates[i]))
		print(table[-1:])
		var = int(input("1 if correct, 0 otherwise: "))
		if var == 0:
			review.append((i,date,l, "wrong totals"))

	print (review)
	return review

def check_columns():
	links_dates = tuple(zip(links, dates))
	review = []
	trues = np.full((15,), True)
	for i, (l, date) in enumerate(links_dates):
		table = pd.read_csv('kerala/new/info_{}.csv'.format(dates[i]))
		checkval = table["Under Observation"] == table["Home Isolation"] + table["Hospitalized"]
		if not (checkval == trues).all():
			review.append((i,date,l, "column sums"))
	print (review)
	return review

# reviews = []
# reviews += check_totals_to_source()
# reviews += check_headers()
# reviews += verify_totals()
# reviews += check_columns()
# print (reviews)
