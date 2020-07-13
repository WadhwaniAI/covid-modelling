import os
import datetime
import pypandoc
from mdutils.mdutils import MdUtils
from mdutils import Html
from pprint import pformat
import pandas as pd
import numpy as np
import pickle
import json

def create_report(predictions_dict, forecast_dict=None, ROOT_DIR='../../reports/'):
    """Creates report (BOTH MD and DOCX) for an input of a dict of predictions for a particular district/region
    The DOCX file can directly be uploaded to Google Drive and shared with the people who have to review

    Arguments:
        predictions_dict {dict} -- Dict of predictions for a particual district/region [NOT ALL Districts]

    Keyword Arguments:
        ROOT_DIR {str} -- the path where the plots and the report would be saved (default: {'../../reports/'})
    """

    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)

    m1_dict = predictions_dict['m1']
    m2_dict = predictions_dict['m2']

    filepath = os.path.join(ROOT_DIR, 'predictions_dict.pkl')
    with open(filepath, 'wb+') as dump:
        pickle.dump(predictions_dict, dump)

    filepath = os.path.join(ROOT_DIR, 'params.json')
    with open(filepath, 'w+') as dump:
        run_params = {
            'm1': m1_dict['run_params'],
            'm2': m2_dict['run_params']
        }
        try:
            del run_params['m1']['model_class']
        except:
            pass
        try:
            del run_params['m2']['model_class']
        except:
            pass
        json.dump(run_params, dump, indent=4)

    m1_dict['all_trials'].to_csv(os.path.join(ROOT_DIR, 'm1-trials.csv'))
    m2_dict['all_trials'].to_csv(os.path.join(ROOT_DIR, 'm2-trials.csv'))

    fitting_date = predictions_dict['fitting_date']
    data_last_date = predictions_dict['data_last_date']
    state = predictions_dict['state']
    dist = predictions_dict['dist']
    filename = os.path.join(ROOT_DIR, f'{state.title()}-{dist.title()}_report_{fitting_date}')
    mdFile = MdUtils(file_name=filename, title=f'{dist.title()} Fits [Based on data until {data_last_date}]')

    mdFile.new_paragraph("---")
    mdFile.new_paragraph(f"State: {state}")
    mdFile.new_paragraph(f"District: {dist}")
    mdFile.new_paragraph(f"Data Source: {predictions_dict['datasource']}")
    mdFile.new_paragraph(f"Data available till: {data_last_date}")
    mdFile.new_paragraph(f"Fitting Date: {fitting_date}")
    mdFile.new_paragraph("---")
    
    mdFile.new_paragraph("Parameter fits were obtained via Bayesian optimization, searching uniformly on the following space of parameter ranges:")
    mdFile.insert_code(pformat(predictions_dict['variable_param_ranges']))

    if m1_dict['plots']['smoothing'] is not None:
        mdFile.new_header(level=1, title=f'SMOOTHING')
        plot_filename = '{}-{}-smoothed-{}.png'.format(state.lower(), dist.lower(), fitting_date)
        plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
        m1_dict['plots']['smoothing'].savefig(plot_filepath)
        mdFile.new_line(mdFile.new_inline_image(text='Smoothing Plot', path=plot_filepath))
        for sentence in m1_dict['smoothing_description'].split('\n'):
            mdFile.new_paragraph(sentence)
        mdFile.new_paragraph("")

    mdFile.new_header(level=1, title=f'FITS')

    mdFile.new_header(level=1, title=f'M1 FIT')
    mdFile.new_header(level=2, title=f'Optimal Parameters')
    mdFile.insert_code(pformat(m1_dict['best_params']))
    mdFile.new_header(level=2, title=f'MAPE Loss Values')
    mdFile.new_paragraph(m1_dict['df_loss'].to_markdown())
    mdFile.new_header(level=2, title=f'M1 Fit Curves')

    plot_filename = '{}-{}-m1-fit-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
    m1_dict['plots']['fit'].savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='M1 Fit Curve', path=plot_filepath))
    mdFile.new_paragraph("")

    mdFile.new_header(level=2, title=f'M1 Sensitivity Curves')

    plot_filename = '{}-{}-m2-sensitivity-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
    m2_dict['plots']['sensitivity'].savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='M1 Fit Sensitivity', path=plot_filepath))
    mdFile.new_paragraph("")

    mdFile.new_header(level=1, title=f'M2 FIT')
    mdFile.new_header(level=2, title=f'Optimal Parameters')
    mdFile.insert_code(pformat(m2_dict['best_params']))
    mdFile.new_header(level=2, title=f'MAPE Loss Values')
    mdFile.new_paragraph(m2_dict['df_loss'].to_markdown())
    mdFile.new_header(level=2, title=f'M2 Fit Curves')

    plot_filename = '{}-{}-m2-fit-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
    m2_dict['plots']['fit'].savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='M2 Fit Curve', path=plot_filepath))
    mdFile.new_paragraph("")

    mdFile.new_header(level=2, title=f'M2 Sensitivity Curves')

    plot_filename = '{}-{}-m2-sensitivity-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
    m2_dict['plots']['sensitivity'].savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='M2 Fit Sensitivity', path=plot_filepath))
    mdFile.new_paragraph("")

    mdFile.new_header(level=2, title=f'Uncertainty')
    mdFile.new_paragraph(f"beta - {m2_dict['beta']}")
    mdFile.new_paragraph(f"beta loss")
    mdFile.insert_code(pformat(m2_dict['beta_loss']))


    mdFile.new_header(level=1, title=f'FORECASTS')
    
    mdFile.new_header(level=2, title=f'FORECAST USING M2 FIT WITH ERROR EVALUATED ON VAL SET OF M1 FIT')
    plot_filename = '{}-{}-forecast-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
    m2_dict['plots']['forecast_best'].savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='Forecast using M2 Best Params', path=plot_filepath))
    mdFile.new_paragraph("")

    mdFile.new_header(level=2, title=f'FORECASTS ON TOTAL INFECTIONS BASED ON TOP k PARAMETER SETS FOR M2 FIT')
    plot_filename = '{}-{}-confirmed-forecast-topk-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
    m2_dict['plots']['forecast_confirmed_topk'].savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='Forecast using M2 of top k params', path=plot_filepath))

    mdFile.new_header(level=2, title=f'FORECASTS ON ACTIVE CASES BASED ON TOP k PARAMETER SETS FOR M2 FIT')
    plot_filename = '{}-{}-active-forecast-topk-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
    m2_dict['plots']['forecast_active_topk'].savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='Forecast using M2 of top k params', path=plot_filepath))
    mdFile.new_paragraph("")

    mdFile.new_header(level=2, title=f'BEST M2 FIT vs 50th DECILE FORECAST')
    plot_filename = '{}-{}-best-50-forecast-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
    m2_dict['plots']['forecast_best_50'].savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='Forecast using best fit, 50th decile params', path=plot_filepath))
    mdFile.new_paragraph("")

    mdFile.new_header(level=2, title=f'BEST M2 FIT vs 80th DECILE FORECAST')
    plot_filename = '{}-{}-best-80-forecast-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
    m2_dict['plots']['forecast_best_80'].savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='Forecast using best fit, 80th decile params', path=plot_filepath))
    mdFile.new_paragraph("")

    mdFile.new_header(level=2, title=f'FORECASTS ON CONFIRMED CASES OF ALL DECILES')
    plot_filename = '{}-{}-confirmed-ptiles-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
    m2_dict['plots']['forecast_confirmed_ptiles'].savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='Forecast on confirmed of all deciles', path=plot_filepath))
    mdFile.new_paragraph("")

    mdFile.new_header(level=2, title=f'FORECASTS ON ACTIVE CASES OF ALL DECILES')
    plot_filename = '{}-{}-active-ptiles-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
    m2_dict['plots']['forecast_active_ptiles'].savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='Forecast on active of all deciles', path=plot_filepath))
    mdFile.new_paragraph("")

    mdFile.new_header(level=2, title=f'FORECASTS ON DAILY CASES OF ALL DECILES')
    plot_filename = '{}-{}-daily-ptiles-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
    m2_dict['plots']['forecast_new_cases_ptiles'].savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='Forecast on daily of all deciles', path=plot_filepath))
    mdFile.new_paragraph("")

    mdFile.new_header(level=1, title="What if Scenarios")
    mdFile.new_header(level=2, title="Linear scale up of testing assuming fixed TPR")
    plot_filename = '{}-{}-whatifs-testing-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
    m2_dict['plots']['whatifs_testing'].savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='Linear scale up of testing assuming fixed TPR', path=plot_filepath))
    mdFile.new_paragraph("")
    mdFile.new_paragraph("---")
    
    mdFile.new_header(level=1, title="Param Values")
    
    mdFile.new_header(level=2, title="Top 10 Trials")
    header = f'| Loss | Params |\n|:-:|-|\n'
    for i, params_dict in enumerate(m2_dict['trials_processed']['params'][:10]):
        tbl = f'| {np.around(m2_dict["trials_processed"]["losses"][:10][i], 2)} |'
        if i == 0:
            tbl = header + tbl
        for key, value in params_dict.items():
            params_dict[key] = np.around(value, 2)
        for lx in pformat(params_dict).split('\n'):
            tbl += f'`{lx}` <br> '
        tbl += '|\n'
        if i != 0:
            tbl += '|-|-|'
        mdFile.new_paragraph(tbl)

    mdFile.new_header(level=2, title="Decile Params")
    header = f'| Decile | Params |\n|:-:|-|\n'
    for i, key in enumerate(m2_dict['deciles'].keys()):
        params_dict = predictions_dict['m2']['deciles'][key]['params']
        tbl = f'| {key} |'
        if i == 0:
            tbl = header + tbl
        for key, value in params_dict.items():
            params_dict[key] = np.around(value, 2)
        for lx in pformat(params_dict).split('\n'):
            tbl += f'`{lx}` <br> '
        tbl += '|\n'
        if i != 0:
            tbl += '|-|-|'
        mdFile.new_paragraph(tbl)

    mdFile.new_header(level=2, title="Decile Loss")
    header = f'| Decile | Loss |\n|:-:|-|\n'
    for i, key in enumerate(m2_dict['deciles'].keys()):
        df_loss = predictions_dict['m2']['deciles'][key]['df_loss']
        tbl = f'| {key} |'
        if i == 0:
            tbl = header + tbl
        for idx in df_loss.index:
            df_loss.loc[idx, 'train'] = np.around(df_loss.loc[idx, 'train'], 2)
        for lx in pformat(df_loss).split('\n'):
            tbl += f'`{lx}` <br> '
        tbl += '|\n'
        if i != 0:
            tbl += '|-|-|'
        mdFile.new_paragraph(tbl)

    mdFile.new_header(level=2, title="Min/Max Loss and Params in Top 10")
    tbl = f'| Param | Min/Max |\n|:-:|-|\n'
    tbl += f'| Loss |'
    vals = [loss for loss in m2_dict['trials_processed']['losses'][:10]]
    minval, maxval = min(vals), max(vals)
    tbl += f'`min: {np.around(minval, 2)}` <br> '
    tbl += f'`max: {np.around(maxval, 2)}` <br> '
    tbl += '|\n'
    for i, key in enumerate(m2_dict['trials_processed']['params'][0]):
        tbl += f'| {key} |'
        vals = [params[key] for params in m2_dict['trials_processed']['params'][:10]]
        minval, maxval = min(vals), max(vals)
        tbl += f'`min: {np.around(minval, 2)}` <br> '
        tbl += f'`max: {np.around(maxval, 2)}` <br> '
        tbl += '|\n'
    tbl += '|-|-|'
    mdFile.new_paragraph(tbl)

    if forecast_dict is not None:
        mdFile.new_header(level=1, title="Uncertainty")
        mdFile.new_header(level=2, title="Percentiles")
        beta = forecast_dict['beta']
        mdFile.new_paragraph(f'Beta: {beta}')
        mdFile.new_paragraph(pd.DataFrame(forecast_dict['decile_params']).to_markdown())
        plot_filename = 'deciles.png'
        plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
        forecast_dict['deciles_plot'].figure.savefig(plot_filepath)
        mdFile.new_line(mdFile.new_inline_image(text='deciles', path=plot_filepath))
        mdFile.new_paragraph("")
        
        mdFile.new_header(level=2, title="What-Ifs")
        plot_filename = 'what-ifs/what-ifs.png'
        plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
        forecast_dict['what-ifs-plot'].savefig(plot_filepath)
        mdFile.new_line(mdFile.new_inline_image(text='what-if scenarios', path=plot_filepath))
        mdFile.new_paragraph("")
        


    # Create a table of contents
    mdFile.new_table_of_contents(table_title='Contents', depth=2)
    mdFile.create_md_file()

    pypandoc.convert_file("{}.md".format(filename), 'docx', outputfile="{}.docx".format(filename))
    pypandoc.convert_file("{}.docx".format(filename), 'pdf', outputfile="{}.pdf".format(filename))
    # TODO: pdf conversion has some issues with order of images, low priority
