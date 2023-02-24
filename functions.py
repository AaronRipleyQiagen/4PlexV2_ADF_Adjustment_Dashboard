import numpy as np
import pandas as pd

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
    return calculate_sensitivity_specificity(simulated_frame)
    
def calculate_current_sensitivity_specificity(data: pd.DataFrame):
    current_frame = data.copy()
    current_frame['Classification'] = classify_result(current_frame, observed_column='Reported Result')
    current_frame[['TP', 'FP', 'TN', 'FN']] = build_classification_array(current_frame)
    return calculate_sensitivity_specificity(current_frame)