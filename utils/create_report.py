import os
import datetime
import pypandoc
from mdutils.mdutils import MdUtils
from mdutils import Html
from pprint import pformat


def create_report(predictions_dict, ROOT_DIR='../../reports/'):
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

    mdFile.new_header(level=1, title=f'M1 FIT')
    mdFile.new_header(level=2, title=f'Optimal Parameters')
    mdFile.insert_code(pformat(predictions_dict['m1']['best_params']))
    mdFile.new_header(level=2, title=f'MAPE Loss Values')
    mdFile.new_paragraph(predictions_dict['m1']['df_loss'].to_markdown())
    mdFile.new_header(level=2, title=f'M1 Fit Curves')

    plot_filename = '{}-{}-m1-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(ROOT_DIR, plot_filename)
    predictions_dict['m1']['ax'].figure.savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='M1 Fit Curve', path=plot_filename))

    mdFile.new_header(level=1, title=f'M2 FIT')
    mdFile.new_header(level=2, title=f'Optimal Parameters')
    mdFile.insert_code(pformat(predictions_dict['m2']['best_params']))
    mdFile.new_header(level=2, title=f'MAPE Loss Values')
    mdFile.new_paragraph(predictions_dict['m2']['df_loss'].to_markdown())
    mdFile.new_header(level=2, title=f'M2 Fit Curves')

    plot_filename = '{}-{}-m2-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(ROOT_DIR, plot_filename)
    predictions_dict['m2']['ax'].figure.savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='M2 Fit Curve', path=plot_filename))

    mdFile.new_paragraph("---")
    
    mdFile.new_header(level=1, title=f'FORECAST USING M2 FIT WITH ERROR EVALUATED ON VAL SET OF M1 FIT')
    plot_filename = '{}-{}-forecast-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    plot_filepath = os.path.join(ROOT_DIR, plot_filename)
    predictions_dict['forecast'].figure.savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text='Forecast using M2', path=plot_filename))
    
    # mdFile.new_header(level=1, title=f'FORECASTS ON TOTAL INFECTIONS BASED ON TOP 10 PARAMETER SETS FOR M2 FIT')
    # plot_filename = '{}-{}-forecast-top10-{}.png'.format(state.lower(), dist.lower(), fitting_date)
    # plot_filepath = os.path.join(ROOT_DIR, plot_filename)
    # predictions_dict['forecast_top10'].figure.savefig(plot_filepath)
    # mdFile.new_line(mdFile.new_inline_image(text='Forecast using M2 of top 10', path=plot_filename))

    # Create a table of contents
    mdFile.new_table_of_contents(table_title='Contents', depth=2)
    mdFile.create_md_file()

    pypandoc.convert_file("{}.md".format(filename), 'docx', outputfile="{}.docx".format(filename))


