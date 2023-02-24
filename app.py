import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_ag_grid as dag
import pandas as pd
import numpy as np
import dash_daq as daq
import base64
import io
from functions import calculate_current_sensitivity_specificity, calculate_simulated_sensitivity_specificity, check_cutoffs



dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.YETI])
app = dash_app.server

def serve_layout():
    
    """
    Define styles
    """

    cutoff_selection_style = {'width': '50%',
                              'display': 'inline-block',
                              'vertical-align': 'middle',
                              'horizontal-align': 'left'}

    sensitivity_specificity_style = {'width': '25%',
                              'display': 'inline-block',
                              'vertical-align': 'middle',
                              'horizontal-align': 'left'}
    customer_fp_style = {'width': '100%',
                          'display': 'inline-block',
                          'vertical-align': 'middle',
                          'horizontal-align': 'left'}

    """
    Build Upload component, message & storage for uploaded data.
    """

    upload_csv = dcc.Upload(
                id='upload-csv',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '50%',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin-left': '25%',
                },
                # Allow multiple files to be uploaded
                multiple=False,
            )

    uploaded_data = dcc.Store(id='uploaded-data', storage_type='memory')
    uploaded_data_msg = html.P(id='uploaded-data-msg')

    
    """
    Build Ct Range Cutoff Components
    """
    ct_range_label = html.Label("Set Min / Max Ct Cycle")
    
    ct_range_marks = {}
    for mark in range(0,50,5):
      ct_range_marks[mark] = {'label':str(mark)}
    valid_ct_window_adjustment = dcc.RangeSlider(0, 50, 1, value=[11, 37], marks=ct_range_marks, id='ct-window-threshold')
    
    min_ep_label = html.Label("Set Minimum Endpoint Fluorescence")
    
    """
    Build Minimum EP Cutoff Components
    """
    ep_marks = {}
    for mark in range(0,5000,500):
      ep_marks[mark] = {'label':str(mark)}

    min_ep_cutoff = dcc.Slider(0, 5000, 100, value=1200, marks=ep_marks, included=False, id='min-ep-threshold')
    
    """
    Build Min Peak Cutoff Components
    """

    min_peak_label = html.Label("Set Minimum Peak Height")

    min_peak_marks = {}
    for mark in range(75,150,5):
      min_peak_marks[mark] = {'label':str(mark)}

    
    min_peak_cutoff = dcc.Slider(75, 150, 1, value=75, marks=min_peak_marks, included=False, id='min-peak-threshold')

    """
    Build EPR Check Ct Threhsold Cutoff Components
    """
    epr_check_ct_threshold_label = html.Label("Set EPR Ct Check Threshold")

    

    epr_check_ct_threshold_marks = {}
    for mark in range(20,40,2):
      epr_check_ct_threshold_marks[mark] = {'label':str(mark)}

    epr_check_ct_threshold_cutoff = dcc.Slider(20, 40, 1, value=30, marks=epr_check_ct_threshold_marks, included=False, id='epr-ct-check-threshold')


    """
    Build EPR Check Threhsold Cutoff Components
    """
    epr_threshold_label = html.Label("Set EPR Check Threshold")

    

    epr_threshold_marks = {}
    for mark in np.arange(1,1.30,0.02):

      epr_threshold_marks[mark] = {'label':str(round(mark, 2))}

    epr_threshold_cutoff = dcc.Slider(1, 1.30, .01, value=1.15, marks=epr_threshold_marks, included=False, id='epr-threshold')
    

    """
    Build Overall EPR Check Threhsold Cutoff Components
    """
    overall_epr_threshold_label = html.Label("Set Overall EPR Check Threshold")

    

    overall_epr_threshold_marks = {}
    for mark in np.arange(1,1.50,0.05):

      overall_epr_threshold_marks[mark] = {'label':str(round(mark, 2))}

    overall_epr_threshold_cutoff = dcc.Slider(1, 1.50, .01, value=1.05, marks=overall_epr_threshold_marks, included=False, id='overall-epr-threshold')


    """
    Build Run Simulation Button, Specimen Types Selection and ADF Setting Selection Storage
    """

    simulation_button = dbc.Button("Run Simulation", id='simulation-button', style=cutoff_selection_style)
    specimen_type_selection_label = html.Label("Filter By Specimen Type")
    specimen_type_selection = dcc.Dropdown(id='specimen-type-selection')
    settings = dcc.Store(id='settings', storage_type='session')
    
    """
    Build Sensitivity / Specificity Gauges 
    """
    
    import dash_daq as daq

    clinical_sensitivity = daq.Gauge(
        color="#9B51E0",
        label='Clinical Sensitivity',
        max=100,
        min=0,
        units="%",
        id='clinical-sensitivity',
        showCurrentValue=True,
        style=sensitivity_specificity_style
    )

    clinical_specificity = daq.Gauge(
        color="#9B51E0",
        label='Clinical Specificity',
        max=100,
        min=0,
        units="%",
        id='clinical-specificity',
        showCurrentValue=True,
        style=sensitivity_specificity_style
    )

    analytical_sensitivity = daq.Gauge(
        color="#9B51E0",
        label='Analytical Sensitivity',
        max=100,
        min=0,
        units="%",
        id='analytical-sensitivity',
        showCurrentValue=True,
        style=sensitivity_specificity_style
    )

    analytical_specificity = daq.Gauge(
        color="#9B51E0",
        label='Analytical Specificity',
        max=100,
        min=0,
        units="%",
        id='analytical-specificity',
        showCurrentValue=True,
        style=sensitivity_specificity_style
        
    )
    
    customer_fp_label = html.Label("Number of customer FPS", style={'width':'20%'})
    customer_fps = daq.GraduatedBar(    
                                        step=1,
                                        size=1000,
                                        showCurrentValue=True,
                                        id='customer-fps',
                                        style=customer_fp_style
                                    )
        
    layout = html.Div([settings,
                       uploaded_data,
                       html.Div([html.H3("Upload CSV File"), uploaded_data_msg,  upload_csv],
                       style={
                              "border": "1px solid black",
                              "padding": "10px"
                              }
                                ),
                       
                       html.Div([
                                html.Div([html.H3("Set ADF Parameter Settings")],style={
                                        "padding": "10px"
                                      }), 
                                html.Div([ct_range_label, valid_ct_window_adjustment], style=cutoff_selection_style),
                                html.Div([min_ep_label, min_ep_cutoff], style=cutoff_selection_style),
                                html.Div([min_peak_label, min_peak_cutoff], style=cutoff_selection_style),
                                html.Div([overall_epr_threshold_label, overall_epr_threshold_cutoff], style=cutoff_selection_style),
                                html.Div([epr_check_ct_threshold_label, epr_check_ct_threshold_cutoff], style=cutoff_selection_style),
                                html.Div([epr_threshold_label, epr_threshold_cutoff], style=cutoff_selection_style),
                                html.Div([specimen_type_selection_label, specimen_type_selection], style=cutoff_selection_style)
                                ],
                                style={
                                        "border": "1px solid black",
                                        "padding": "10px"
                                      }
                                ),
                       html.Div([clinical_sensitivity,
                                 clinical_specificity,
                                 analytical_sensitivity,
                                 analytical_specificity,
                                 customer_fp_label, customer_fps
                                 ],style={
                                        "border": "1px solid black",
                                        "padding": "10px"
                                      }),
                      ])
    
    return layout

dash_app.layout = serve_layout

@dash_app.callback([Output('uploaded-data', 'data'),
                    Output('uploaded-data-msg', 'children')],
                   Input('upload-csv', 'contents'))
def store_uploaded_data(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        dashboard_data = pd.read_csv(io.BytesIO(decoded))
        dashboard_data['Reported Result'] = dashboard_data['Far Red Target Localized Result'].replace({'TargetNotAmplified':'NEG', 'TargetAmplified':'POS'})
        dashboard_data_valid = dashboard_data[dashboard_data['Reported Result'].isin(['POS', 'NEG'])]
        dashboard_data_valid = dashboard_data_valid[dashboard_data_valid['Far Red Target Expected Result']!='Exclude']
        dashboard_data_valid['Expected Result'] = np.where(dashboard_data_valid['Far Red Target Expected Result']=='NEG', "NEG", "POS")
        data_dict = dashboard_data_valid.to_dict('records')
        return data_dict, "Successfully uploaded data for {} valid samples".format(str(len(dashboard_data_valid)))
    else:
      return {}, ''

@dash_app.callback(Output('specimen-type-selection', 'options'),
              Input('uploaded-data', 'data'), prevent_initial_call=True)
def store_uploaded_data(uploaded_data):
  if uploaded_data:
    dataframe = pd.DataFrame.from_dict(uploaded_data)
    specimen_type_options = {}
    for specimen_type  in dataframe['Target Setting Specimen Type'].unique():
      specimen_type_options[specimen_type] = specimen_type
    return specimen_type_options
  else:
    return {}

@dash_app.callback(Output('settings', 'data'),
                   [Input('ct-window-threshold', 'value'),
                   Input('min-ep-threshold', 'value'),
                   Input('min-peak-threshold', 'value'),
                   Input('overall-epr-threshold', 'value'),
                   Input('epr-ct-check-threshold', 'value'),
                   Input('epr-threshold', 'value'),
                   Input('specimen-type-selection','value')
                   ], prevent_intial_call=True)
def get_settings(ct_window, min_ep, min_peak, overall_epr, epr_ct_check, epr, specimen_type):
  settings = {'minimum-peak-cyle-threshold':ct_window[0],
              'maximum-peak-cycle-threshold':ct_window[1],
              'minimum-ep-threshold':min_ep,
              'minimum-peak-threshold':min_peak,
              'overall-epr-check-threshold':overall_epr,
              'epr-check-ct-threshold':epr_ct_check,
              'epr-check-threshold':epr,
              'specimen-type':specimen_type}
  return settings

@dash_app.callback([Output('clinical-sensitivity','value'),
                    Output('clinical-specificity','value'),
                    Output('analytical-sensitivity','value'),
                    Output('analytical-specificity','value')],
                   [Input('settings', 'data'),
                    State('uploaded-data', 'data')],
                    prevent_initial_call=True)
def update_sensitivity_specificity_kpis(settings, uploaded_data):
  if uploaded_data:
    dataframe = pd.DataFrame.from_dict(uploaded_data)
    if settings['specimen-type']:
      dataframe = dataframe[dataframe['Target Setting Specimen Type']==settings['specimen-type']]
    dataframe_clincial = dataframe[dataframe['Data Source']=='Clinical']
    dataframe_analytical = dataframe[dataframe['Data Source']=='Analytical']
    clinical_sensitivity, clinical_specificity = calculate_simulated_sensitivity_specificity(dataframe_clincial, settings, 'Far Red')
    analytical_sensitivity, analytical_specificity = calculate_simulated_sensitivity_specificity(dataframe_analytical, settings, 'Far Red')
    return clinical_sensitivity*100, clinical_specificity*100, analytical_sensitivity*100, analytical_specificity*100
  else:
    return dash.no_update

@dash_app.callback(Output('customer-fps', 'max'),
                   Output('customer-fps', 'value'),
                   [Input('settings', 'data'),
                    State('uploaded-data', 'data')])
def get_customer_fps(settings, uploaded_data):
  if uploaded_data:
    
    dataframe = pd.DataFrame.from_dict(uploaded_data)
    if settings['specimen-type']:
      dataframe = dataframe[dataframe['Target Setting Specimen Type']==settings['specimen-type']]
    dataframe_customer = dataframe[dataframe['Data Source']=='Customer']
    dataframe_customer['Simulated Target Result'] = check_cutoffs(dataframe_customer, settings, 'Far Red')

    return len(dataframe_customer), len(dataframe_customer[dataframe_customer['Simulated Target Result']!='NEG'])
  else:

    return dash.no_update



    

if __name__ == '__main__':
    
    dash_app.run_server(debug=True)