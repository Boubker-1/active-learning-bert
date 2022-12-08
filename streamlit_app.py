# Active Learning framework using Streamlit
# Algorithms used for fine tuning and Active Learning can be found at:
# Transformers Meet Active Learning: Less Data, Better Performance - https://towardsdatascience.com/transformers-meet-active-learning-less-data-better-performance-4cf931517ff6
# Fine-Tuning BERT for Text Classification - https://towardsdatascience.com/fine-tuning-bert-for-text-classification-54e7df642894#e7cb
# My work was to combine both blogs and managing the labelling system using Streamlit

import streamlit as st
import pandas as pd
import re, torch
from al_functions import *
from time import sleep

if "page" not in st.session_state:
    st.session_state.page = 0

torch.cuda.empty_cache()

def nextpage(): st.session_state.page += 1
def add_datasets(dataset, name):
    st.session_state[name] = dataset
    st.session_state.page += 1
def prevpage(): st.session_state.page -= 1
def restart():
    torch.cuda.empty_cache()
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state.page = 0

def clean_text(text):
    text = " ".join(text.split("\n"))
    no_mentions = re.sub('([\u0600-\u06FF]+)@', '', text)
    no_underscore = re.sub(r'_', ' ', no_mentions)
    lines = no_underscore.split("\n")
    filtered_lines = [re.sub(r'[^\u0600-\u06FF ]', '', line) for line in lines if line != ""]
    filtered =  '\n'.join(filtered_lines);
    no_diacritics = re.sub(r'[^\u0621-\u063A\u0641-\u064A\u0660-\u0669a-zA-Z 0-9]', '', filtered)
    no_punctuations = re.sub(r'،؛؟«»!', '', no_diacritics)
    clean = re.sub(r'(.)\1+', r'\1', no_punctuations)
    return clean.lstrip().rstrip()

def keep_columns(df, to_keep):
    cols = df.keys().tolist()
    return df.drop([col for col in cols if col not in to_keep], axis=1)

def transform_train(df, text_col, label_col, labels):
    df = keep_columns(df, [text_col, label_col])
    df = df.rename(columns={text_col: 'text', label_col:'label'}, inplace=False)
    for i, label in enumerate(labels):
        df.label = df.label.replace({label:i})
    df.text = df.text.apply(clean_text)
    return df

def transform_pool(df, text_col):
    df = keep_columns(df, [text_col])
    df = df.rename(columns={text_col: 'text'}, inplace=False)
    df['label'] = [-1 for _ in range(len(df['text']))]
    df.text = df.text.apply(clean_text)
    return df

def initiate_model():
    st.session_state['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.session_state['model'], st.session_state['optimizer'] = create_model(st.session_state['bert'], num_labels=len(st.session_state['train_labels']))
    st.session_state['results'] = []
    st.session_state['model'].cuda()
    st.session_state['pool_dataloader'], st.session_state['train_dataloader'] = data_prep(st.session_state['pool'], st.session_state['train'], st.session_state['bert'], st.session_state['batch_size'])
    query_from_pool()

def query_from_pool():
    logits = []
    for _ in range(st.session_state['epochs']):
        training(st.session_state['model'], st.session_state['optimizer'], st.session_state['train_dataloader'], st.session_state['device'])
        epoch_logits = return_logits(st.session_state['model'], st.session_state['pool_dataloader'], st.session_state['device'])
        for logit in epoch_logits:
            logits.append(logit)
    st.session_state['results'].append(eval_active_learning(st.session_state['model'], logits, st.session_state['train_dataloader'], st.session_state['device']))
    st.session_state['query_idx'] = torch.topk(calculate_entropy(np.asarray(logits)), st.session_state['query_samples']).indices.tolist()

def annotate():
    st.session_state['samples_labels'] = {}
    for j in st.session_state['query_idx']:
        st.session_state['samples_labels'][j] = st.radio(st.session_state['pool']['text'][j], options=np.asarray(st.session_state['train_labels']))
    st.write(st.session_state['samples_labels'])

def update_train_pool():
    text_col = [st.session_state['pool']['text'][k] for k in list(st.session_state['samples_labels'].keys())]
    label_col = [v for v in list(st.session_state['samples_labels'].values())]
    query_samples = pd.DataFrame({'text':text_col, 'label':label_col})
    for i, label in enumerate(st.session_state['train_labels']):
        query_samples.label = query_samples.label.replace({label:i})
    st.session_state['train'] = pd.concat([st.session_state['train'], query_samples])
    st.session_state['pool'] = st.session_state['pool'].drop(st.session_state['query_idx'])
    st.session_state['pool'] = st.session_state['pool'].reset_index(drop=True)
    query_from_pool()

placeholder = st.empty()

if st.session_state.page==0:
    with placeholder.container():
        st.title('Active Learning with AraBERT')
        st.subheader('**This app is built to help selecting the best unlabelled text data for annotations using Active Learning!**')
        bert = st.selectbox('Select the model (large model requires more GPU resources)', ('AraBERT Base', 'AraBERT Large'))
        if bert=='AraBERT Base':
            st.session_state['bert'] = 'aubmindlab/bert-base-arabertv2'
        else:
            st.session_state['bert'] = 'aubmindlab/bert-large-arabertv2'

        st.write('Please enter the path (or URL) for both train and unlabelled datasets. Make sure that both are in CSV format and have the same separating character!')
        col1, col2, col3 = st.columns(3)
        with col1:
            train = st.text_input('Train')
        with col2:
            pool = st.text_input('Unlabelled')
        with col3:
            sep = st.text_input('Separator')

        if train and pool and sep:
            train_dataset = pd.read_csv(train, sep=sep)
            pool_dataset = pd.read_csv(pool, sep=sep)
            st.markdown('From here we need to select the text data and label columns to keep from both sets. For convinience, please make sure that they both have the same columns with the same name')
            col1, col2 = st.columns(2)
            with col1:
                text_col = st.selectbox('Text data', train_dataset.keys().tolist())
            with col2:
                label_col = st.selectbox('Label', train_dataset.keys().tolist())
            pool_text = st.selectbox('Unlabelled text data', pool_dataset.keys().tolist())
            confirm = st.checkbox('Confirm')
            if confirm:
                st.session_state['train_labels'] = train_dataset[label_col].unique().tolist()
                st.write('We detected the following labels:', st.session_state['train_labels'])
                st.write('Each label will be give a unique number corresponding to its index')
                accept = st.checkbox('Accept')
                if accept:
                    st.session_state['train'] = transform_train(train_dataset, text_col, label_col, st.session_state['train_labels'])
                    st.session_state['pool'] = transform_pool(pool_dataset, pool_text)
                    st.write('Please verify the data training and labelling on the next page!')
                    st.button("Next", on_click=nextpage, disabled=False)

elif st.session_state.page==1:
    with placeholder.container():
        st.title('Active Learning with AraBERT')
        st.subheader('**This app is built to help selecting the best unlabelled text data for annotations using Active Learning!**')
        st.write("New training set after encoding.")
        st.dataframe(st.session_state['train'])
        st.write("New unlabelled set. Note that we added a new column for Active Learning logic. -1 is a dummy label and doesn't affect the process whatsoever!")
        st.dataframe(st.session_state['pool'])
        st.write('If all the data is fine, please hit "Next" button to proceed into Active Learning process or "Reset" to reset all data')
        col1, col2 = st.columns(2)
        with col1:
            st.button("Next", on_click=nextpage, disabled=False)
        with col2:
            st.button("Reset", on_click=restart, disabled=False)

elif st.session_state.page==2:
    with placeholder.container():
        st.title('Active Learning with AraBERT')
        st.subheader('**This app is built to help selecting the best unlabelled text data for annotations using Active Learning!**')
        st.write('Now, we will attempt our Active Learning process! Please enter the following parameters:')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state['epochs'] = st.number_input("Epochs", format='%d', step=1, min_value=1)
        with col2:
            st.session_state['batch_size'] = st.number_input("Batch Size", format='%d', step=1, min_value=1)
        with col3:
            st.session_state['query_samples'] = st.number_input("Number of query samples", format='%d', step=1, min_value=1)
        
        st.write('Click "Initiate" to setup our model and query our first samples!')
        init = st.button("Initiate", on_click=initiate_model)
        if init:
            st.write('Model and first query initiated! Proceed to labelling by clicking "Next"')
            st.session_state['querying'] = True
            col1, col2 = st.columns(2)
            with col1:
                st.button("Next", on_click=nextpage, disabled=False)
            with col2:
                st.button("Reset", on_click=restart, disabled=False)

elif st.session_state.page==3:
    with placeholder.container():
        st.title('Active Learning with AraBERT')
        st.subheader('**This app is built to help selecting the best unlabelled text data for annotations using Active Learning!**')
        st.write('Active Learning process has started! You can repeat the process as much as you want by pressing "Update And Query" button! When done, press "Stop" to check training results!')
        sub_placeholder = st.empty()
        with sub_placeholder.container():
            sub_placeholder = st.empty()
            if st.session_state['querying']:
                current_step = len(st.session_state['results'])
                st.write("Query number " + str(current_step))
                st.write('Please select correct annotations to add for unlabelled pool:')
                annotate()
                col1, col2 = st.columns(2)
                with col1:
                    update = st.button("Update And Query", on_click=update_train_pool)
                with col2:
                    stop = st.button("Stop")
                current_results = st.session_state['results'][current_step-1]
                st.write('Current training accuracy: %.2f' %(current_results[0][0]))
                st.write('Current training precision: %.2f' %(current_results[1][0]))
                st.write('Current training recall: %.2f' %(current_results[2][0]))
                if stop:
                    st.session_state['querying'] = False
            else:
                st.write('All querying process is done!')
                col1, col2 = st.columns(2)
                with col1:
                    st.button("Next", on_click=nextpage, disabled=False)
                with col2:
                    st.button("Reset", on_click=restart, disabled=False)

elif st.session_state.page==4:
    with placeholder.container():
        st.title('Active Learning with AraBERT')
        st.subheader('**This app is built to help selecting the best unlabelled text data for annotations using Active Learning!**')
        st.write('After training, let\'s see how the training is evolving by queries')
        steps = len(st.session_state['results'])
        results = st.session_state['results']
        accuracy, precision, recall= [], [], []
        for i in range(len(results)):
            accuracy.append(results[i][0][0])
            precision.append(results[i][1][0])
            recall.append(results[i][2][0])
        results_data = pd.DataFrame({'Query':[i+1 for i in range(steps)], 'Accuracy':accuracy, 'Precision':precision, 'Recall':recall})
        st.dataframe(results_data)
        chart_data = results_data.drop(columns=['Query'])
        st.line_chart(chart_data)
        st.write('You can save your new annotated data into a csv file by clicking "Save Training Set"!')
        col1, col2 = st.columns(2)
        with col1:
            save = st.button('Save Training Set')
        with col2:
            st.button("Reset", on_click=restart, disabled=False)
        if save:
            st.session_state['train'].to_csv('al_train.csv', index=False)
            st.write('File saved successfully!')
