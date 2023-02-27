import dash
from dash import Input, Output, State
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_ag_grid as dag
import dash_daq as daq
import pandas as pd
import numpy as np
import base64
import io

dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.YETI], suppress_callback_exceptions=True)
app = dash_app.server
dash_app.title = 'NeuMoDx ADF Tuner'

def check_cutoffs(data:pd.DataFrame, settings:dict, target:str):
  target_result_frame = data.copy()
  ##double check RP spec for EPR check Ct threshold implementation.
  target_result_frame['Simulated Target Result'] = np.where((
                                                  (target_result_frame[target + ' Target Max Peak Height'].round(2)>settings['minimum-peak-threshold'])&
                                                  (target_result_frame[target + ' Target End Point Fluorescence'].round(2)>settings['minimum-ep-threshold'])&
                                                  (target_result_frame[target+' Target Ct'].round(2)<=(settings['maximum-peak-cycle-threshold'])+0.5)&
                                                  (target_result_frame[target+' Target EPR'].round(2)>=settings['overall-epr-check-threshold'])&
                                                  (
                                                   (target_result_frame[target+' Target Ct']>settings['epr-check-ct-threshold'])|
                                                   ((target_result_frame[target+' Target Ct']<=settings['epr-check-ct-threshold'])&(target_result_frame[target+' Target EPR'].round(2)>settings['epr-check-threshold']))
                                                  )),'POS','NEG')
  return target_result_frame['Simulated Target Result']

def classify_result(data: pd.DataFrame, expected_column='Expected Result', observed_column='Simulated Target Result', pos_result="POS", neg_result="NEG"):
    """
    A function used to grade results and return TP, FP, TN, FN based on alignment logic between expected and observed columns
    """
    classify_dataframe = data.copy()
    classify_dataframe['Classification'] = np.nan
    classify_dataframe['Classification'] = np.where(((classify_dataframe[expected_column]==pos_result)&
                                                     (classify_dataframe[observed_column]==pos_result)),
                                                     'TP',
                                                     classify_dataframe['Classification'])
    classify_dataframe['Classification'] = np.where(((classify_dataframe[expected_column]==neg_result)&
                                                     (classify_dataframe[observed_column]==pos_result)),
                                                     'FP',
                                                     classify_dataframe['Classification'])
    classify_dataframe['Classification'] = np.where(((classify_dataframe[expected_column]==neg_result)&
                                                     (classify_dataframe[observed_column]==neg_result)),
                                                     'TN',
                                                     classify_dataframe['Classification'])
    classify_dataframe['Classification'] = np.where(((classify_dataframe[expected_column]==pos_result)&
                                                     (classify_dataframe[observed_column]==neg_result)),
                                                     'FN',
                                                     classify_dataframe['Classification'])
    
    return classify_dataframe['Classification']
    
def build_classification_array(data: pd.DataFrame, classification_column='Classification'):
    """
    A function used to build an array of values that can be used to create a set of 4 boolean columns that describe whether a result is a TP, FP, TN, FN
    """

    classify_array_dataframe  = data.copy()
    classify_array_dataframe['TP'] = np.where(classify_array_dataframe[classification_column]=='TP', 1, 0)
    classify_array_dataframe['FP'] = np.where(classify_array_dataframe[classification_column]=='FP', 1, 0)
    classify_array_dataframe['TN'] = np.where(classify_array_dataframe[classification_column]=='TN', 1, 0)
    classify_array_dataframe['FN'] = np.where(classify_array_dataframe[classification_column]=='FN', 1, 0)

    return classify_array_dataframe[['TP', 'FP', 'TN', 'FN']]
    
def calculate_sensitivity_specificity(data: pd.DataFrame, TP_column='TP', FP_column="FP", TN_column='TN', FN_column='FN'):
    
    sensitivity_specificity_dataframe = data.copy()

    sensitivity_specificity_dataframe_grouped = sensitivity_specificity_dataframe[[TP_column, FP_column, TN_column, FN_column]].agg('sum')

    sensitivity = sensitivity_specificity_dataframe_grouped[TP_column] / (sensitivity_specificity_dataframe_grouped[TP_column] + sensitivity_specificity_dataframe_grouped[FN_column])
    specificity = sensitivity_specificity_dataframe_grouped[TN_column] / (sensitivity_specificity_dataframe_grouped[TN_column] + sensitivity_specificity_dataframe_grouped[FP_column])

    return sensitivity, specificity

def calculate_simulated_sensitivity_specificity(data: pd.DataFrame, settings: dict, target: str):
    simulated_frame = data.copy()
    simulated_frame['Simulated Target Result'] = check_cutoffs(simulated_frame, settings, target)
    simulated_frame['Classification'] = classify_result(simulated_frame)
    simulated_frame[['TP', 'FP', 'TN', 'FN']] = build_classification_array(simulated_frame)
    return calculate_sensitivity_specificity(simulated_frame), simulated_frame
    
def calculate_current_sensitivity_specificity(data: pd.DataFrame):
    current_frame = data.copy()
    current_frame['Classification'] = classify_result(current_frame, observed_column='Reported Result')
    current_frame[['TP', 'FP', 'TN', 'FN']] = build_classification_array(current_frame)
    return calculate_sensitivity_specificity(current_frame)

def serve_layout():
    
    """
    Define styles
    """

    cutoff_selection_style = {'width': '50%',
                              'display': 'inline-block',
                              'vertical-align': 'middle',
                              'horizontal-align': 'left'}

    sensitivity_specificity_style = {'width': '20%',
                              'display': 'inline-block',
                              'vertical-align': 'Top',
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
    for mark in range(100,130,2):

      epr_threshold_marks[(mark/100)] = {'label':str(round((mark/100), 2))}

    epr_threshold_cutoff = dcc.Slider(1, 1.30, .01, marks=epr_threshold_marks, value=1.15, included=False, id='epr-threshold')
    

    """
    Build Overall EPR Check Threhsold Cutoff Components
    """
    overall_epr_threshold_label = html.Label("Set Overall EPR Check Threshold")

    

    overall_epr_threshold_marks = {}
    for mark in range(100,150,5):

      overall_epr_threshold_marks[(mark/100)] = {'label':str(round((mark/100), 2))}

    overall_epr_threshold_cutoff = dcc.Slider(1, 1.50, .01, marks=overall_epr_threshold_marks, value=1.05, included=False, id='overall-epr-threshold')


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
    

    clinical_sensitivity = daq.Gauge(
        color={"gradient":True,"ranges":{"red":[0,85],"yellow":[85,95],"green":[95,100]}},
        label='Clinical Sensitivity',
        max=100,
        min=0,
        units="%",
        id='clinical-sensitivity',
        showCurrentValue=True,
        style=sensitivity_specificity_style
    )

    clinical_specificity = daq.Gauge(
        color={"gradient":True,"ranges":{"red":[0,85],"yellow":[85,95],"green":[95,100]}},
        label='Clinical Specificity',
        max=100,
        min=0,
        units="%",
        id='clinical-specificity',
        showCurrentValue=True,
        style=sensitivity_specificity_style
    )

    analytical_sensitivity = daq.Gauge(
        color={"gradient":True,"ranges":{"red":[0,85],"yellow":[85,95],"green":[95,100]}},
        label='Analytical Sensitivity',
        max=100,
        min=0,
        units="%",
        id='analytical-sensitivity',
        showCurrentValue=True,
        style=sensitivity_specificity_style
    )

    analytical_specificity = daq.Gauge(
        color={"gradient":True,"ranges":{"red":[0,85],"yellow":[85,95],"green":[95,100]}},
        label='Analytical Specificity',
        max=100,
        min=0,
        units="%",
        id='analytical-specificity',
        showCurrentValue=True,
        style=sensitivity_specificity_style
        
    )
    
    customer_fps = daq.Tank(    
                              label='Customer Reported False Positives',
                              showCurrentValue=True,
                              id='customer-fps',
                              style=sensitivity_specificity_style
                          )
    
    clinical_sensitivity_impact = daq.LEDDisplay(
                                          id="clinical-sensitivity-impact",
                                          label="Impact to Clinical Sensitivity",
                                          value='0.00',
                                          color="green",
                                          style=sensitivity_specificity_style
                                      )
    clinical_specificity_impact = daq.LEDDisplay(
                                          id="clinical-specificity-impact",
                                          label="Impact to Clinical Specificity",
                                          value='0.00',
                                          color="green",
                                          style=sensitivity_specificity_style
                                      )
    analytical_sensitivity_impact = daq.LEDDisplay(
                                          id="analytical-sensitivity-impact",
                                          label="Impact to Analytical Sensitivity",
                                          value='0.00',
                                          color="green",
                                          style=sensitivity_specificity_style
                                      )                                  
    analytical_specificity_impact = daq.LEDDisplay(
                                          id="analytical-specificity-impact",
                                          label="Impact to Analytical Specificity",
                                          value='0.00',
                                          color="green",
                                          style=sensitivity_specificity_style
                                      )
    
    customer_fp_impact = daq.LEDDisplay(
                                          id="customer-fps-impact",
                                          label="Impact to Customer Reported FPs",
                                          value='0',
                                          color="green",
                                          style=sensitivity_specificity_style
                                      )
    
    """
    Assemble the Card Body for Summary Results
    """

    summary_content = dbc.Card(

    dbc.CardBody(
            [
                html.Div([clinical_sensitivity,
                                 clinical_specificity,
                                 analytical_sensitivity,
                                 analytical_specificity,
                                 customer_fps,
                                 clinical_sensitivity_impact,
                                 clinical_specificity_impact,
                                 analytical_sensitivity_impact,
                                 analytical_specificity_impact,
                                 customer_fp_impact
                                 ],style={
                                        "border": "1px solid black",
                                        "padding": "5px"
                                      }),
            ]
        )
    )    
    


    """
    Build the Card Body for Affected Sample Results
    """

    affected_samples = dag.AgGrid(
                enableEnterpriseModules=True,
                # columnDefs=initial_columnDefs,
                # rowData=intial_data,
                columnSize="sizeToFit",
                defaultColDef=dict(
                    resizable=True,
                ),
                rowSelection='single',
                # setRowId="id",
                id='affected-samples-table'
            )

    affected_samples_content = dbc.Card(
      dbc.CardBody(
        [
          affected_samples
        ]
        
      )
    )

    """
    Build Data Review Tabs Component 
    """
    data_review_tabs = dbc.Tabs(
      children=[
          dbc.Tab(summary_content, label="Data Summary", id='data-summary'),
          dbc.Tab(affected_samples_content, label="Affected Samples", id='affected-samples')
      ], id='summary-tab'
    )

    """
    Build data storage components for affected samples data.
    """
    clincial_affected_samples_data = dcc.Store(id='clinical-affected-samples-data', storage_type='session')
    analytical_affected_samples_data = dcc.Store(id='analytical-affected-samples-data', storage_type='session')
    customer_fps_affected_samples_data = dcc.Store(id='customer-fps-samples-data', storage_type='session')

    return html.Div(children=[settings,
                       uploaded_data,
                       clincial_affected_samples_data,
                       analytical_affected_samples_data,
                       customer_fps_affected_samples_data,
                       html.Div(children=[html.H3("Upload CSV File"), uploaded_data_msg,  upload_csv],
                       style={
                              "border": "1px solid black",
                              "padding": "10px"
                              }
                                ),
                       
                       html.Div(children=[
                                html.Div(children=[html.H3("Set ADF Parameter Settings")],style={
                                        "padding": "10px"
                                      }), 
                                html.Div(children=[ct_range_label, valid_ct_window_adjustment], style=cutoff_selection_style),
                                html.Div(children=[min_ep_label, min_ep_cutoff], style=cutoff_selection_style),
                                html.Div(children=[min_peak_label, min_peak_cutoff], style=cutoff_selection_style),
                                html.Div(children=[overall_epr_threshold_label, overall_epr_threshold_cutoff], style=cutoff_selection_style),
                                html.Div(children=[epr_check_ct_threshold_label, epr_check_ct_threshold_cutoff], style=cutoff_selection_style),
                                html.Div(children=[epr_threshold_label, epr_threshold_cutoff], style=cutoff_selection_style),
                                html.Div(children=[specimen_type_selection_label, specimen_type_selection], style=cutoff_selection_style)
                                ],
                                style={
                                        "border": "1px solid black",
                                        "padding": "10px"
                                      }
                                ),
                       data_review_tabs,
                       
                      ])

dash_app.layout = serve_layout

@dash_app.callback([Output('uploaded-data', 'data'),
                    Output('uploaded-data-msg', 'children')],
                    Input('upload-csv', 'contents'), prevent_initial_call=True)
def store_uploaded_data(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        dashboard_data = pd.read_csv(io.BytesIO(decoded))
        dashboard_data['Reported Result'] = dashboard_data['Far Red Target Localized Result'].replace({'TargetNotAmplified':'NEG', 'TargetAmplified':'POS'})
        dashboard_data_valid = dashboard_data[dashboard_data['Reported Result'].isin(['POS', 'NEG'])]
        dashboard_data_valid = dashboard_data_valid[~dashboard_data_valid['Far Red Target Expected Result'].isin(['Exclude','LPOS'])]
        dashboard_data_valid['Expected Result'] = np.where(dashboard_data_valid['Far Red Target Expected Result']=='NEG', "NEG", "POS")
        data_dict = dashboard_data_valid.to_dict('records')
        return data_dict, "Successfully uploaded data for {} valid samples".format(str(len(dashboard_data_valid)))
    else:
      return dash.no_update

@dash_app.callback(Output('specimen-type-selection', 'options'),
              Input('uploaded-data', 'data'), prevent_initial_call=True)
def store_uploaded_data(uploaded_data):
  if uploaded_data:
    dataframe = pd.DataFrame.from_dict(uploaded_data)
    specimen_type_options = {}
    for specimen_type  in dataframe['Target Setting Specimen Type'].unique():
      specimen_type_options[specimen_type] = specimen_type
    specimen_type_options['All'] = 'All'
    return specimen_type_options
  else:
    return dash.no_update

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
  if ct_window:
    settings = {'minimum-peak-cyle-threshold':ct_window[0],
                'maximum-peak-cycle-threshold':ct_window[1],
                'minimum-ep-threshold':min_ep,
                'minimum-peak-threshold':min_peak,
                'overall-epr-check-threshold':overall_epr,
                'epr-check-ct-threshold':epr_ct_check,
                'epr-check-threshold':epr,
                'specimen-type':specimen_type}

    return settings
  else:
    return dash.no_update

@dash_app.callback([Output('clinical-sensitivity','value'),
                    Output('clinical-sensitivity-impact', 'value'),
                    Output('clinical-specificity','value'),
                    Output('clinical-specificity-impact', 'value'),
                    Output('analytical-sensitivity','value'),
                    Output('analytical-sensitivity-impact', 'value'),
                    Output('analytical-specificity','value'),
                    Output('analytical-specificity-impact', 'value'),
                    Output('customer-fps', 'max'),
                    Output('customer-fps', 'value'),
                    Output('customer-fps-impact', 'value'),
                    Output('clinical-affected-samples-data', 'data'),
                    Output('analytical-affected-samples-data','data'),
                    Output('customer-fps-samples-data', 'data')],
                   [Input('settings', 'data'),
                    State('uploaded-data', 'data')],
                    prevent_initial_call=True)
def update_sensitivity_specificity_fps_kpis(settings, uploaded_data):
  if uploaded_data:
    dataframe = pd.DataFrame.from_dict(uploaded_data)
    if settings['specimen-type']:
      if settings['specimen-type']!='All':
        dataframe = dataframe[dataframe['Target Setting Specimen Type']==settings['specimen-type']]
    dataframe_clincial = dataframe[dataframe['Data Source']=='Clinical']
    dataframe_analytical = dataframe[dataframe['Data Source']=='Analytical']
    
    original_clinical_sensitivity, original_clinical_specificity = calculate_current_sensitivity_specificity(dataframe_clincial)
    original_analytical_sensitivity, original_analytical_specificity = calculate_current_sensitivity_specificity(dataframe_analytical)
    (clinical_sensitivity, clinical_specificity), dataframe_clinical = calculate_simulated_sensitivity_specificity(dataframe_clincial, settings, 'Far Red')
    (analytical_sensitivity, analytical_specificity), dataframe_analytical = calculate_simulated_sensitivity_specificity(dataframe_analytical, settings, 'Far Red')
    
    dataframe_clinical_changes = dataframe_clinical[dataframe_clinical['Simulated Target Result']!=dataframe_clinical['Reported Result']]
    dataframe_analytical_changes = dataframe_analytical[dataframe_analytical['Simulated Target Result']!=dataframe_analytical['Reported Result']]

    dataframe_customer = dataframe[dataframe['Data Source']=='Customer']
    dataframe_customer['Simulated Target Result'] = check_cutoffs(dataframe_customer, settings, 'Far Red')
    dataframe_customer_changes = dataframe_customer[dataframe_customer['Simulated Target Result']!=dataframe_customer['Reported Result']]
    return clinical_sensitivity*100, ((clinical_sensitivity-original_clinical_sensitivity)*100).round(2), clinical_specificity*100, ((clinical_specificity-original_clinical_specificity)*100).round(2), analytical_sensitivity*100, ((analytical_sensitivity-original_analytical_sensitivity)*100).round(2), analytical_specificity*100, ((analytical_specificity-original_analytical_specificity)*100).round(2), len(dataframe_customer), len(dataframe_customer[dataframe_customer['Simulated Target Result']!='NEG']), len(dataframe_customer[dataframe_customer['Simulated Target Result']!='NEG']) - len(dataframe_customer), dataframe_clinical_changes.to_dict('records'), dataframe_analytical_changes.to_dict('records'), dataframe_customer_changes.to_dict('records')
  else:
    return dash.no_update

@dash_app.callback(Output('clinical-sensitivity-impact','color'),
              Input('clinical-sensitivity-impact','value'), prevent_initial_call=True)
def update_clinical_sensitivity_color(value):
  if value < 0:
    return 'red'
  else:
    return 'green'

@dash_app.callback(Output('clinical-specificity-impact','color'),
              Input('clinical-specificity-impact','value'), prevent_initial_call=True)
def update_clinical_sensitivity_color(value):
  if value < 0:
    return 'red'
  else:
    return 'green'

@dash_app.callback(Output('analytical-sensitivity-impact','color'),
              Input('analytical-sensitivity-impact','value'), prevent_initial_call=True)
def update_clinical_sensitivity_color(value):
  if value < 0:
    return 'red'
  else:
    return 'green'

@dash_app.callback(Output('analytical-specificity-impact','color'),
              Input('analytical-specificity-impact','value'), prevent_initial_call=True)
def update_clinical_sensitivity_color(value):
  if value < 0:
    return 'red'
  else:
    return 'green'

@dash_app.callback(Output('customer-fps-impact','color'),
                   Input('customer-fps-impact','value'), prevent_initial_call=True)
def update_clinical_sensitivity_color(value):
  if value <= 0:
    return 'green'
  else:
    return 'red'

@dash_app.callback([Output('affected-samples-table', 'rowData'),
                    Output('affected-samples-table', 'columnDefs')],
                   [Input('clinical-affected-samples-data', 'data'),
                    Input('analytical-affected-samples-data', 'data'),
                    Input('customer-fps-samples-data', 'data'),
                   ],prevent_initial_call=True)
def get_affected_sample_results(clinical_samples, analytical_samples, customer_fps):
  clinical_samples_dataframe = pd.DataFrame.from_dict(clinical_samples)
  analytical_samples_dataframe = pd.DataFrame.from_dict(analytical_samples)
  customer_fps_dataframe = pd.DataFrame.from_dict(customer_fps)
  all_affected_samples = pd.concat([clinical_samples_dataframe,analytical_samples_dataframe,customer_fps_dataframe])
  print(all_affected_samples)
  visable_columns = ['Data Source', 'Sample ID', 'Protocol', 'Target Setting Specimen Type', 'Expected Result', 'Reported Result', 'Simulated Target Result', 'Notes']
  column_definitions = []
  for column in all_affected_samples.columns:
      if column == 'Data Source' or column == 'Protocol' or column == 'Target Setting Specimen Type':
        column_definitions.append(
              {"headerName": column, "field": column, "rowGroup": True, "filter": True})
      elif column in visable_columns:
          column_definitions.append(
              {"headerName": column, "field": column, "filter": True})
      else:
          column_definitions.append(
              {"headerName": column, "field": column, "filter": True, "hide": True})

  return all_affected_samples.to_dict('records'),column_definitions

if __name__ == '__main__':
    
    dash_app.run_server(debug=True)