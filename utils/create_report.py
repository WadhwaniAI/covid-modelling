import os
from mdutils.mdutils import MdUtils
from mdutils import Html
from pprint import pformat

def create_report(path, dist, state, date, datasource, m1, m2, overall):
    filename = os.path.join(path, f'{dist.title()}_report_{date}')
    mdFile = MdUtils(file_name=filename, title=f'{dist.title()} Official Data Fits [Based on data until {date}]')

    mdFile.new_paragraph(f"Data Source: {datasource}")
    mdFile.new_paragraph("---")
    
    mdFile.new_paragraph("Parameter fits were obtained via Bayesian optimization, searching uniformly on the following space of parameter ranges:")
    mdFile.insert_code(pformat(overall['variable_param_ranges']))

    mdFile.new_header(level=1, title=f'M1 FIT')
    mdFile.new_header(level=2, title=f'Optimal Parameters')
    mdFile.insert_code(pformat(m1['best_params']))
    mdFile.new_header(level=2, title=f'MAPE Loss Values')
    mdFile.new_paragraph(m1['loss_df'])
    mdFile.new_header(level=2, title=f'M1 Fit Curves')
    mdFile.new_line(mdFile.new_inline_image(text='M1 Fit Curve', path=m1['plot_path']))

    mdFile.new_header(level=1, title=f'M2 FIT')
    mdFile.new_header(level=2, title=f'Optimal Parameters')
    mdFile.insert_code(pformat(m2['best_params']))
    mdFile.new_header(level=2, title=f'MAPE Loss Values')
    mdFile.new_paragraph(m2['loss_df'])
    mdFile.new_header(level=2, title=f'M2 Fit Curves')
    mdFile.new_line(mdFile.new_inline_image(text='M2 Fit Curve', path=m2['plot_path']))

    mdFile.new_paragraph("---")
    
    mdFile.new_header(level=1, title=f'FORECAST USING M2 FIT WITH ERROR EVALUATED ON VAL SET OF M1 FIT')
    mdFile.new_line(mdFile.new_inline_image(text='M2_val_M1', path=overall['m2_val_m1_path']))
    mdFile.new_header(level=1, title=f'FORECASTS ON ACTIVE INFECTIONS BASED ON TOP 10 PARAMETER SETS FOR M2 FIT')
    # mdFile.new_line(mdFile.new_inline_image(text='active infections', path=overall['top_10_active_path']))
    mdFile.insert_code(pformat(overall['top_10_trials']))

    # Create a table of contents
    mdFile.new_table_of_contents(table_title='Contents', depth=2)
    mdFile.create_md_file()


