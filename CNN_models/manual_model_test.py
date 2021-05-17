import os
import pandas as pd
import numpy as np
import spacy
from spacy.util import minibatch, compounding
import random
import re
import sys
import argparse
from tabulate import tabulate

def manual_model_test(input_data, version=1):
    """
    Takes a string input of a complaint from the CFPB and will
    score the complaint using all product models for the specified version#.
    Output will be the a print of the final predicted product for the 
    complaint was filed against. This is defined to be the highest predicted
    score among all model scores by product.
    
    Argumenst:
    input: str
    version: int
    """
    #  Load saved trained model
    model_directory = os.getcwd()
    product_models = [model for model in os.listdir(model_directory) if model[-1:]==str(version)]
    loaded_models = [spacy.load(os.path.join(model_directory, model))
                     for model in product_models
                    ]
    if __name__ != '__main__':
        print(f'Review text: \n{input_data}')
    
    i=0
    prediction_dict = {}
    
    table_headers = ['Product Classifier', 'Prediction', 'Score']
    
    iter_list = []
    for model in loaded_models:
        intra_iter_list = []
        m = re.match(f"(\w+)_simple_cnn_model_artifacts_v\d", product_models[i])
        if m:
            current_product = m.group(1)
            intra_iter_list.append(current_product)
        
        prediction_dict[current_product] = {}
        
        # Generate prediction
        parsed_text = model(input_data)
        # Determine prediction to return
        if parsed_text.cats["Y"] > parsed_text.cats["N"]:
            prediction = "Y"
            score = parsed_text.cats["Y"]
        else:
            prediction = "N"
            score = -1*parsed_text.cats["N"]
        
        prediction_dict[current_product]['prediction'] = prediction
        prediction_dict[current_product]['score'] = score
        
        intra_iter_list.append(prediction)
        intra_iter_list.append(score)
        
        iter_list.append(intra_iter_list)
        i+=1
    
    candidate_predictions = {product: prod_dict['score']
                             for product, prod_dict in prediction_dict.items() 
                             if prod_dict['prediction']=='Y'
                            }
    highest_score_idx = list(candidate_predictions.values()).index(max(list(candidate_predictions.values())))
    predicted_product = list(candidate_predictions.keys())[highest_score_idx]
    print('='*51)
    print(tabulate(iter_list, headers=table_headers, tablefmt='orgtbl'))
    print(f"\n!!!!!!!!!!========== AND THE WINNER IS!!!!! ==============!!!!!!!!!!\n\n{' '*21}{predicted_product}\n")

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Arguments for the manual_model_test cli app.')
    my_parser.add_argument('-i', '--input',
                           type=str, help='A string containing a complaint to be classified by product.',
                           required=True)
    my_parser.add_argument('-v', '--version',
                           type=str, help='Specifies which version of the models to classify the input string using.',
                           required=False)
    args = my_parser.parse_args()
    input_string = vars(args)['input']
    print('='*50)
    print('manual_model_test.py has been called via cli. Will classify input_string provided and will classify what product_group it is predicted to belong to.\n\nPowered by spaCy!\n')
    print(f'Input String:\n{input_string}')
    print('='*50)
    manual_model_test(input_string)
    print('='*10 + ' Classification complete ' + '='*10)