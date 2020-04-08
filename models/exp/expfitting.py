import numpy as np 
from scipy import special
from scipy.optimize import curve_fit, leastsq
from matplotlib import pyplot as plt 
from numpy import exp, log, format_float_positional
import pandas as pd
from dfisman_r0 import load_data, get_state
import sys
from inspect import signature
from sklearn.metrics import r2_score, mean_squared_log_error

def poly(x, a, b): 
    return a*(x**b)

def poly_str(a, b, fname=False): 
	if fname:
	    return "a*(x**b)"
	with np.printoptions(precision=4, suppress=True):
	    return "{a}*(x**{b})".format(
	    	a=format_float_positional(a, precision=4),
	    	b=format_float_positional(b, precision=4)
	    )

def exponential(x, a, b): 
    return a*exp(b*x)

def exponential_str(a, b, fname=False): 
	if fname:
	    return "a*exp(b*x)"

	with np.printoptions(precision=4, suppress=True):
	    return "{a}*exp({b}*x)".format(
	    	a=format_float_positional(a, precision=4),
	    	b=format_float_positional(b, precision=4)
	    )
def logreg(x, a, b, c):
	return a/(1+exp(-b*(x-c)))

def logreg_str(a, b, c, fname=False):
	if fname:
		return "a/(1+exp(-b*(x-c)))"
	with np.printoptions(precision=4, suppress=True):
	    return "{a}/(1+exp(-{b}*(x-{c})))".format(
	    	a=format_float_positional(a, precision=4),
	    	b=format_float_positional(b, precision=4),
	    	c=format_float_positional(c, precision=4)
	    )

# error function cdf of the normal distribution
def erf(t, a, b, c):
    return 0.5*c*(special.erf(a*(t - b)) + 1.0)

def erf_str(a, b, c, fname=False):
	if fname:
		return "0.5*p*(special.erf(a*(t - b)) + 1.0)"
	with np.printoptions(precision=4, suppress=True):
	    return "0.5*{c}*(special.erf({a}*(t - {b})) + 1.0)".format(
	    	a=format_float_positional(a, precision=4),
	    	b=format_float_positional(b, precision=4),
	    	c=format_float_positional(c, precision=4)
	    )

def model_I_t(R_i, t):
	# I(t) = sum_{i=0}^t R_i^i,
	return sum([R_i[i]**i for i in range(int(t))])

def model_I_t_helper(R_i_val, i):
	return R_i_val**i

def model_I_t_str():
	return "I(t) = sum_{i=0}^t R_i^i"

def fit_leastsq(p0, x, y, function): # https://stackoverflow.com/a/21844726

	log_diff = lambda params,x,y: log(y) - log(function(x, *params))

	errfunc = log_diff

	pfit, pcov, infodict, errmsg, success = \
    	leastsq(errfunc, p0, args=(x, y), full_output=1, epsfcn=0.0001)
	if (len(y) > len(p0)) and pcov is not None:
	    s_sq = (errfunc(pfit, x, y)**2).sum()/(len(y)-len(p0))
	    pcov = pcov * s_sq
	else:
	    pcov = np.inf
	error = [] 
	for i in range(len(pfit)):
	    try:
	      error.append(np.absolute(pcov[i][i])**0.5)
	    except:
	      error.append( 0.00 )
	pfit_leastsq = pfit
	perr_leastsq = np.array(error) 
	return pfit_leastsq, perr_leastsq 

def curve_fitting(funcs, df, xcol, ycol, fname):
	# least squares log diff loss function
	# funcs: list(functions)
		# each function must also have a function_str method that takes in parameters and fname=False
		# see above for examples
	x, y = df[xcol], df[ycol]
	colors = ['r--', 'g--', 'c--', 'm--', 'y--', 'k--']

	for i, func in enumerate(funcs):
		# popt, pcov = curve_fit(func, x, y)
		# popt, pcov = leastsq(func=log_diff, x0=(1.,1.), args=(x,y))
		# perr = np.sqrt(np.diag(pcov))
		popt, perr = fit_leastsq((len(signature(func).parameters)-1)*[1.], x, y, func)

		print("{} coefficients:".format(func.__name__)) 
		print(popt) 
		print("covariance of coefficients:") 
		print(perr) 
		print()


		strfunc = getattr(sys.modules[__name__], func.__name__ + "_str")
		plt.yscale("log")
		plt.grid()
		plt.xlabel("Days")
		plt.ylabel("Cases")
		plt.plot(x, y, 'b-', label='data')
		plt.plot(x, func(x, *popt), 'r-', label='fit: {}: {}'.format(func.__name__, strfunc(*popt)))
		if i < 2:
			plt.title("{} Cases fit to {}".format(fname, strfunc(1,1,fname=True)))
			plt.plot(x, func(x, a=popt[0] + perr[0], b=popt[1] + perr[1]), 'g--', label='fit: {}: {}'.format(func.__name__, strfunc(a=popt[0] + perr[0], b=popt[1] + perr[1])))
			plt.plot(x, func(x, a=popt[0] - perr[0], b=popt[1] - perr[1]), 'g--', label='fit: {}: {}'.format(func.__name__, strfunc(a=popt[0] - perr[0], b=popt[1] - perr[1])))
			plt.plot([], [], ' ', label="std dev for a: {} b: {}".format(format_float_positional(perr[0], precision=4), format_float_positional(perr[1], precision=4)))

		else:
			plt.title("{} Cases fit to {}".format(fname, strfunc(1,1,1,fname=True)))
			plt.plot(x, func(x, a=popt[0] + perr[0], b=popt[1] + perr[1], c=popt[2] + perr[2]), 'g--', label='fit: {}: {}'.format(func.__name__, strfunc(a=popt[0] + perr[0], b=popt[1] + perr[1], c=popt[2] + perr[2])))
			plt.plot(x, func(x, a=popt[0] - perr[0], b=popt[1] - perr[1], c=popt[2] - perr[2]), 'g--', label='fit: {}: {}'.format(func.__name__, strfunc(a=popt[0] - perr[0], b=popt[1] - perr[1], c=popt[2] - perr[2])))
			plt.plot([], [], ' ', label="std dev for a: {} b: {} c: {}".format(format_float_positional(perr[0], precision=4), format_float_positional(perr[1], precision=4), format_float_positional(perr[2], precision=4)))

		# popt[0] +/- perr[0], popt[1] +/- perr[1]

		# plt.plot(x, ans, '--', color ='blue', label ="optimized data") 
		plt.legend() 
		plt.savefig('output/{}_fitting_{}.png'.format(fname, func.__name__))
		# plt.show() 
		plt.clf() 

def same_plot(funcs, df, xcol, ycol, fname):
	# least squares log diff loss function
	# funcs: list(functions)
		# each function must also have a function_str method that takes in parameters and fname=False
		# see above for examples
	x, y = df[xcol], df[ycol]
	colors_solid = ['r-', 'g-', 'c-', 'm-', 'y-', 'k-']
	colors_dash = ['r--', 'g--', 'c--', 'm--', 'y--', 'k--']
	plt.yscale("log")
	plt.grid()
	plt.xlabel("Days")
	plt.ylabel("Cases")
	plt.plot(x, y, 'b-', label='data')
		
	for i, func in enumerate(funcs):
		# popt, pcov = curve_fit(func, x, y)
		# popt, pcov = leastsq(func=log_diff, x0=(1.,1.), args=(x,y))
		# perr = np.sqrt(np.diag(pcov))
		popt, perr = fit_leastsq((len(signature(func).parameters)-1)*[1.], x, y, func)

		print("{} coefficients:".format(func.__name__)) 
		print(popt) 
		print("covariance of coefficients:") 
		print(perr) 
		print()

		ypred = func(x, *popt)
		r2 = r2_score(y, ypred)
		print ("r2: {}".format(r2))
		msle = mean_squared_log_error(y, ypred)
		print ("msle: {}".format(msle))
		strfunc = getattr(sys.modules[__name__], func.__name__ + "_str")
		plt.plot(x, ypred, colors_solid[i], label='fit: {}: {}'.format(func.__name__, strfunc(*popt)))
		plt.title("{} Cases".format(fname))
		print 
		plt.plot(x, func(x, a=popt[0] + perr[0], b=popt[1] + perr[1], c=popt[2] + perr[2]), colors_dash[i], label='fit: {}: {}'.format(func.__name__, strfunc(a=popt[0] + perr[0], b=popt[1] + perr[1], c=popt[2] + perr[2])))
		plt.plot(x, func(x, a=popt[0] - perr[0], b=popt[1] - perr[1], c=popt[2] - perr[2]), colors_dash[i], label='fit: {}: {}'.format(func.__name__, strfunc(a=popt[0] - perr[0], b=popt[1] - perr[1], c=popt[2] - perr[2])))
		plt.plot([], [], ' ', label="std dev for a: {} b: {} c: {}".format(format_float_positional(perr[0], precision=4), format_float_positional(perr[1], precision=4), format_float_positional(perr[2], precision=4)))

	plt.legend() 
	plt.savefig('output/{}_fitting_sameplot.png'.format(fname))
	# plt.show() 
	plt.clf() 


if __name__ == '__main__':
	
# --------------------------------------------
# Set up Data
	ny = pd.read_csv('../../data/data/ny_google.csv')
	x = pd.Series([i+1 for i in range(len(ny))])
	dates = ny["date"] # x - time in days
	y = ny["cases"] # number of cases on day x
	Ri = ny["Rt_smooth"]
	ri = ny["r"]
	ri_smooth = ny["r_smooth"]
	nyxy = pd.concat([x, y, dates, Ri, ri, ri_smooth], axis=1)

	bbmp = pd.read_csv('../../data/data/bbmp.csv')
	x = pd.Series([i+1 for i in range(len(bbmp))])
	dates = bbmp["Date"] # x - time in days
	y = bbmp["Cumulative Cases Til Date"] # number of cases on day x
	bbmpxy = pd.concat([x, dates, y], axis=1)

# --------------------------------------------
# Time/Cases

	funcs = [poly, exponential, logreg, erf]
	# curve_fitting(funcs, nyxy, 0, 'cases', 'NY')
	# curve_fitting(funcs, bbmpxy, 0, 'Cumulative Cases Til Date', 'BBMP')
	same_plot([logreg, erf], bbmpxy, 0, 'Cumulative Cases Til Date', 'BBMP')
# --------------------------------------------
# R_t/Cases

	# # for i in range(5):
	# for i in range(len(x)):
	# 	xy.at[i, 'y_intermediate_Ri'] = model_I_t_helper(Ri[i], i)
	# 	xy.at[i, 'y_intermediate_ri'] = model_I_t_helper(ri[i], i)
	# 	xy.at[i, 'y_intermediate_ri_smooth'] = model_I_t_helper(ri_smooth[i], i)

	# xy["y_pred_Ri"] = xy['y_intermediate_Ri'].cumsum()
	# xy["y_pred_ri"] = xy['y_intermediate_ri'].cumsum()
	# xy["y_pred_ri_smooth"] = xy['y_intermediate_ri_smooth'].cumsum()

	# print (xy)
	# # ok so the pred value blows up because of high r_i values initially, we need a better method of calculating for this to be meaningful

	# def setup_r_graphs(title_str):
	# 	plt.yscale("log")
	# 	plt.xlabel("Time: Days since {}".format(dates[0]))
	# 	plt.ylabel("Cases")
	# 	plt.title(model_I_t_str() + title_str)
	# 	plt.grid()

	# setup_r_graphs(", where R_i is using Fisman's spreadsheet formula")
	# plt.plot(x, y, 'b-', label='data')
	# plt.plot(x, xy['y_pred_Ri'], 'g--', label="I(t) = sum_{i=0}^t R_i^i")
	# plt.savefig('output/Ri.png')
	# plt.legend() 
	# # plt.show() 
	# plt.clf() 

	# setup_r_graphs(", where R_i=cases/prev_cases")
	# plt.plot(x, y, 'b-', label='data')
	# plt.plot(x, xy['y_pred_ri'], 'g--', label="I(t) = sum_{i=0}^t R_i^i")
	# plt.savefig('output/small_ri.png')
	# plt.legend() 
	# # plt.show() 
	# plt.clf() 

	# setup_r_graphs(", where R_i=sliding 5 day avg of cases/prev_cases")
	# plt.plot(x, y, 'b-', label='data')
	# plt.plot(x, xy['y_pred_ri_smooth'], 'g--', label="I(t) = sum_{i=0}^t R_i^i")
	# plt.savefig('output/small_ri_smoothed.png')
	# plt.legend() 
	# # plt.show() 
	# plt.clf() 

