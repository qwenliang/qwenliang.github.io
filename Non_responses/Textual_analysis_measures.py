# textual analysis: Tone Uncertainty Forward Fog

import pandas as pd
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from readability import Readability
import textstat
import re
import argparse

def measures(df, lm_path, forward_path, output_path):
    # Load Loughran-McDonald dictionary and forward-looking words
    LM = pd.read_csv(lm_path)
    forward = pd.read_csv(forward_path, header=None)

    # Create lists from the dictionaries
    positive = LM[LM.Positive > 0].Word.to_list()
    negative = LM[LM.Negative > 0].Word.to_list()
    uncertain = LM[LM.Uncertainty > 0].Word.to_list()
    f_l = forward[0].to_list()
    tokenizer = RegexpTokenizer(r'\w+')
    tones = []
    uncertains = []
    forwards = []
    fogs1 = []
    fogs2 = []
    fogs3 =[]
    
    for j in range(len(df)):
        text = df.joint_qa.iloc[j]
        #res = [word.strip(string.punctuation) for word in text.split() if word.strip(string.punctuation).isalnum()]
        res = tokenizer.tokenize(text)
        pos_count = 0
        neg_count = 0
        un_count = 0
        for word in res:
            if word.upper() in positive:
                pos_count+=1
            if word.upper() in negative:
                neg_count+=1
            if word.upper() in uncertain:
                un_count+=1
        try:
            tone = (pos_count - neg_count)/(pos_count+neg_count)
            unc = un_count/len(res)
        except ZeroDivisionError:
            tone = 0
            unc = 0 
        fll = 0
        for fword in f_l:
            matches = list(re.finditer(re.escape(fword.upper()), text.upper()))
            #temp = [i for i in range(len(text)) if text.upper().startswith(fword.upper(), i)] 
            fll += len(matches)
            #print(fword)
        #for fword in f_l:
        #    temp = [i for i in range(len(text)) if text.upper().startswith(fword.upper(), i)] 
        #    fll += len(temp)
        tones.append(tone)
        uncertains.append(unc)
        try:
            r = Readability(text)
            gf = r.gunning_fog()
            fogs1.append(gf.score)
            fogs2.append(gf.grade_level)
            gf_score = textstat.gunning_fog(text)
            fogs3.append(gf_score)
            
        except:
            fogs1.append(-1)
            fogs2.append('None')
            fogs3.append(-1)
        
        forwards.append(fll/len(res))
        
    df['Tone'] = tones
    df['Uncertainty'] = uncertains
    df['Forward'] = forwards
    df['Fog_Score'] = fogs1
    df['Fog_Level'] = fogs2
    df['Fog_textstat']=fogs3

    # Drop the componenttext column if it exists
    df = df.drop(columns=['joint_qa'], errors='ignore')
    # Drop the componenttext column if it exists
    df = df.drop(columns=['pairid_str'], errors='ignore')
    # Drop the componenttext column if it exists
    df = df.drop(columns=['mostimportantdateutc'], errors='ignore')
    # Save the updated DataFrame to a CSV file
    df.to_csv(output_path, index=False)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process text data for tone, uncertainty, and readability.")
    parser.add_argument('--input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('--lm_csv', type=str, help='Path to the Loughran-McDonald dictionary CSV file')
    parser.add_argument('--forward_txt', type=str, help='Path to the forward-looking words text file')
    parser.add_argument('--output_csv', type=str, help='Path to the output CSV file')

    # Parse the arguments
    args = parser.parse_args()

    # Load the initial DataFrame from the input CSV file
    df = pd.read_csv(args.input_file)
    
    # Call the measures function with the provided arguments
    measures(df, args.lm_csv, args.forward_txt, args.output_csv)

if __name__ == "__main__":
    main()
