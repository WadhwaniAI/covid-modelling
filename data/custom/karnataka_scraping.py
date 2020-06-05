import tabula
import logging
import pandas as pd
import numpy as np
logging.getLogger('PDFBox').setLevel(logging.WARNING)

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
