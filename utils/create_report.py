import os
import datetime
import pypandoc
from mdutils.mdutils import MdUtils
from mdutils import Html
from pprint import pformat
import pickle
import json

def create_report(predictions_dict, ROOT_DIR='../../reports/'):
    """Creates report (BOTH MD and DOCX) for an input of a dict of predictions for a particular district/region
    The DOCX file can directly be uploaded to Google Drive and shared with the people who have to review

    Arguments:
        predictions_dict {dict} -- Dict of predictions for a particual district/region [NOT ALL Districts]

    Keyword Arguments:
        ROOT_DIR {str} -- the path where the plots and the report would be saved (default: {'../../reports/'})
    """

    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)

    filepath = os.path.join(ROOT_DIR, 'predictions_dict.pkl')
    with open(filepath, 'wb+') as dump:
        pickle.dump(predictions_dict, dump)

    filepath = os.path.join(ROOT_DIR, 'params.json')
    with open(filepath, 'w+') as dump:
        run_params = {
            'm1': predictions_dict['m1']['run_params'],
            'm2': predictions_dict['m2']['run_params']
        }
        json.dump(run_params, dump)

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

    if 'smoothing_plot' in predictions_dict['m1'].keys():
        mdFile.new_header(level=1, title=f'SMOOTHING')
        plot_filename = '{}-{}-smoothed-{}.png'.format(state.lower(), dist.lower(), fitting_date)
        plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
        predictions_dict['m1']['smoothing_plot'].figure.savefig(plot_filepath)
        mdFile.new_line(mdFile.new_inline_image(text='M1 Fit Curve', path=plot_filepath))
        mdFile.new_paragraph("")

    mdFile.new_header(level=1, title=f'M1 FIT')
    mdFile.new_header(level=2, title=f'Optimal Parameters')
    mdFile.insert_code(pformat(predictions_dict['m1']['best_params']))
    mdFile.new_header(level=2, title=f'MAPE Loss Values')
    mdFile.new_paragraph(predictions_dict['m1']['df_loss'].to_markdown())
    mdFile.new_header(level=2, title=f'M1 Fit Curves')

    plot_filename = '{}-{}-m1-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
    predictions_dict['m1']['ax'].figure.savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='M1 Fit Curve', path=plot_filepath))
    mdFile.new_paragraph("")

    mdFile.new_header(level=1, title=f'M2 FIT')
    mdFile.new_header(level=2, title=f'Optimal Parameters')
    mdFile.insert_code(pformat(predictions_dict['m2']['best_params']))
    mdFile.new_header(level=2, title=f'MAPE Loss Values')
    mdFile.new_paragraph(predictions_dict['m2']['df_loss'].to_markdown())
    mdFile.new_header(level=2, title=f'M2 Fit Curves')

    plot_filename = '{}-{}-m2-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
    predictions_dict['m2']['ax'].figure.savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='M2 Fit Curve', path=plot_filepath))
    mdFile.new_paragraph("")

    mdFile.new_paragraph("---")
    
    forecast_dict = predictions_dict['forecast']
    mdFile.new_header(level=1, title=f'FORECAST USING M2 FIT WITH ERROR EVALUATED ON VAL SET OF M1 FIT')
    plot_filename = '{}-{}-forecast-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
    forecast_dict['forecast'].figure.savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='Forecast using M2', path=plot_filepath))
    mdFile.new_paragraph("")
    

    if len(forecast_dict.keys()) > 0:
        mdFile.new_header(level=1, title=f'FORECASTS ON TOTAL INFECTIONS BASED ON TOP k PARAMETER SETS FOR M2 FIT')
        plot_filename = '{}-{}-confirmed-forecast-topk-{}.png'.format(state.lower(), dist.lower(), fitting_date)
        plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
        forecast_dict['forecast_confirmed_topk'].figure.savefig(plot_filepath)
        mdFile.new_line(mdFile.new_inline_image(text='Forecast using M2 of top k', path=plot_filepath))
        mdFile.new_header(level=1, title=f'FORECASTS ON ACTIVE CASES BASED ON TOP k PARAMETER SETS FOR M2 FIT')
        plot_filename = '{}-{}-active-forecast-topk-{}.png'.format(state.lower(), dist.lower(), fitting_date)
        plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), plot_filename)
        forecast_dict['forecast_active_topk'].figure.savefig(plot_filepath)
        mdFile.new_line(mdFile.new_inline_image(text='Forecast using M2 of top k', path=plot_filepath))
        mdFile.new_paragraph("")
        
        mdFile.new_header(level=2, title="Top 10 Trials")
        header = f'| Loss | Params |\n|:-:|-|\n'
        for i, params_dict in enumerate(forecast_dict['params'][:10]):
            tbl = f'| {forecast_dict["losses"][:10][i]} |'
            if i == 0:
                tbl = header + tbl
            for l in pformat(params_dict).split('\n'):
                tbl += f'`{l}` <br> '
            tbl += '|\n'
            if i != 0:
                tbl += '|-|-|'
            mdFile.new_paragraph(tbl)
    
        mdFile.new_header(level=2, title="Min/Max Loss and Params in Top 100")
        tbl = f'| Param | Min/Max |\n|:-:|-|\n'
        tbl += f'| Loss |'
        vals = [loss for loss in forecast_dict['losses'][:100]]
        minval, maxval = min(vals), max(vals)
        tbl += f'`min: {minval}` <br> '
        tbl += f'`max: {maxval}` <br> '
        tbl += '|\n'
        for i, key in enumerate(forecast_dict['params'][0]):
            tbl += f'| {key} |'
            vals = [params[key] for params in forecast_dict['params'][:100]]
            minval, maxval = min(vals), max(vals)
            tbl += f'`min: {minval}` <br> '
            tbl += f'`max: {maxval}` <br> '
            tbl += '|\n'
        tbl += '|-|-|'
        mdFile.new_paragraph(tbl)
    
    # Create a table of contents
    mdFile.new_table_of_contents(table_title='Contents', depth=2)
    mdFile.create_md_file()

    pypandoc.convert_file("{}.md".format(filename), 'docx', outputfile="{}.docx".format(filename))
    # pypandoc.convert_file("{}.md".format(filename), 'tex', format='md', outputfile="{}.pdf".format(filename))
    # TODO: pdf conversion has some issues with order of images, low priority

