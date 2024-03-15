import pandas as pd
from difflib import SequenceMatcher
import re

def column_dropper(df, columns_to_drop):
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    return df

def indicate_nulls(df, null_check_columns):
    # Create a copy of the dataframe to avoid modifying the original one
    result_df = df.copy()
    # Loop through the list of columns
    for col in null_check_columns:
        # Create a new column indicating whether each row in the original column is null
        new_col_name = f"{col}_is_null"
        result_df[new_col_name] = result_df[col].isnull().astype(int)
    return result_df

# Define the fuzzy search function
def fuzzy_search(search_key, text, strictness):
    lines = text.split("\n")
    for line in lines:
        words = line.split()
        for word in words:
            similarity = SequenceMatcher(None, word.lower(), search_key.lower())
            if similarity.ratio() > strictness:
                return True
    return False

# Function to count appearances of text in each cell of a column and check if greater than 0
def dynamic_check_text(df, target_col, text_column, function_type):
    # Define a lambda function that performs the operation based on the function_type
    if function_type == 'check':
        # Returns 1 if the text is found in the target column, else 0
        process_text = lambda row: 1 if len(re.findall(re.escape(str(row[text_column])), str(row[target_col]), flags=re.IGNORECASE)) > 0 else 0
    elif function_type == 'count':
        # Returns the count of occurrences of the text in the target column
        process_text = lambda row: len(re.findall(re.escape(str(row[text_column])), str(row[target_col]), flags=re.IGNORECASE))
    elif function_type == 'fuzzy_check':
        # Uses the fuzzy_search function to check for the presence of the text with typos
        process_text = lambda row: 1 if fuzzy_search(str(row[text_column]), str(row[target_col]), strictness=0.6) else 0
        
    # Apply this lambda function to each row and return the result
    return df.apply(process_text, axis=1)

def check_columns_on_query(df, target_columns, text_column, function_type='check'):
    # Apply the function to each target column and create new columns
    for column in target_columns:
        new_col_name = f'check_{column}'
        df[new_col_name] = dynamic_check_text(df, column, text_column, function_type)

    return df

def check_columns_on_text(df, column_of_interest, high_occurrence_set=None, relevant_set=None, irrelevant_set=None):
    # Helper function to ensure the column value is iterable for the check
    # If the value is a string or non-iterable, it converts it into a list
    # Otherwise, it returns the value as is (assuming it's already a list or similar iterable)
    def ensure_iterable(value):
        if isinstance(value, str) or not hasattr(value, '__iter__'):
            return [value]  # Convert single elements to a list
        return value  # Return iterable as is

    # Iterate through each text in the high_occurrence_set
    for high_occurrence in high_occurrence_set:
        # Create a new column for each text, indicating if the text is in the target column
        df[f'{column_of_interest}_contains_overlap_{high_occurrence}'] = df[column_of_interest].apply(
            lambda x: 1 if high_occurrence in ensure_iterable(x) else 0)

    # Add similar checks for relevant and irrelevant sets
    df[f'{column_of_interest}_contains_relevant'] = df[column_of_interest].apply(
        lambda x: 1 if any(element in ensure_iterable(x) for element in relevant_set) else 0)
    df[f'{column_of_interest}_contains_irrelevant'] = df[column_of_interest].apply(
        lambda x: 1 if any(element in ensure_iterable(x) for element in irrelevant_set) else 0)

    # Return the modified DataFrame
    return df

def dim_binner(df, column, bins, labels):
    df['temp_column'] = pd.to_numeric(df[column], errors='coerce')
    binned_non_nans = pd.cut(df.loc[df[column].notna(), 'temp_column'], bins=bins, labels=labels, right=False)

    # Create a new column for the binned data
    binned_column_name = column + '_binned'
    
    # Place the binned data back into the DataFrame
    df.loc[df[column].notna(), binned_column_name] = binned_non_nans

    df.drop('temp_column', axis=1, inplace=True)
    
    return df

def check_columns_on_dim(df, column_of_interest, relevant_set=None, irrelevant_set=None):
    # Make sure the sets are not None to avoid 'in None' TypeError
    if relevant_set is None:
        relevant_set = set()
    if irrelevant_set is None:
        irrelevant_set = set()
    
    # Add similar checks for relevant and irrelevant sets
    # Modify lambda functions to handle NaN: pd.notna(x) checks if x is not NaN
    df[f'{column_of_interest}_contains_relevant'] = df[column_of_interest].apply(
        lambda x: 1 if pd.notna(x) and x in relevant_set else 0)
    df[f'{column_of_interest}_contains_irrelevant'] = df[column_of_interest].apply(
        lambda x: 1 if pd.notna(x) and x in irrelevant_set else 0)

    # No need to fill NaN after adjusting the lambda functions, as they now handle NaN implicitly
    return df

from src.eda import null_columns_checker, column_investigator, find_unique_values, tree_path_investigator
from src.eda import dimension_investigator

# for final training
## df needs to have y values
def add_new_features(df, train_or_val='train', val_inputs = []):
    if train_or_val != 'train':
        [null_check_columns, null_columns_to_drop, 
                tpf_rel_feature_potential,  tpf_irrel_feature_potential, 
                tpo_rel_feature_potential,  tpo_irrel_feature_potential, 
                tt_rel_feature_potential,  tt_irrel_feature_potential, 
                h_rel_feature_potential,  h_irrel_feature_potential, 
                s_rel_feature_potential,  s_irrel_feature_potential, 
                ss_rel_feature_potential,  ss_irrel_feature_potential, 
                w_rel_feature_potential,  w_irrel_feature_potential, 
                c_rel_feature_potential,  c_irrel_feature_potential, 
                st_rel_feature_potential,  st_irrel_feature_potential
                ] = val_inputs
        
    
    if train_or_val == 'train':
        _, null_check_columns, null_columns_to_drop = null_columns_checker(df, 0.9, 1.1)

    try:
        # Remove columns with mostly null
        df = column_dropper(df, null_columns_to_drop)
        print('Successfully dropping fully Null columns!\n')
    except Exception as e:
         print("Error:", e)
         
    try:
        # Indicate nulls for partial null columns
        df = indicate_nulls(df, null_check_columns)
        print('Successfully indicating partially Null columns!\n')
    except Exception as e:
         print("Error:", e)
         
    try:
        # Text-based columns analysis
        df = check_columns_on_query(df, ['title', 'alt', 'text'], 'query', 'fuzzy_check')
        print('Successfully adding text-based features!\n')
    except Exception as e:
         print("Error:", e)
         
    try:
        # URL-based columns analysis
        df = check_columns_on_query(df, ['url_page', 'src', 'source'], 'query', 'check')
        print('Successfully adding url-based features!\n')
    except Exception as e:
         print("Error:", e)
         
    try:
        # Tree-path first analysis
        df['tree_path_first'] = df['tree_path'].apply(lambda x: x[0] if len(x) else None)
        if train_or_val == 'train':
            _, tpf_rel_feature_potential, tpf_irrel_feature_potential = column_investigator(df, 'tree_path_first', 1, 0.5, 2)

        df = check_columns_on_text(df, 'tree_path_first', ['a', 'div'], tpf_rel_feature_potential, tpf_irrel_feature_potential)
        print('Successfully adding first tree-path feature!\n')
    except Exception as e:
         print("Error:", e)
         
    try:
        # Tree-path Overall analysis
        text_set = find_unique_values(df, 'tree_path')
        if train_or_val == 'train':
            _, tpo_rel_feature_potential, tpo_irrel_feature_potential = tree_path_investigator(df, text_set, 'tree_path', 5, 0.5, 2)

        high_occurrence = ['a', 'li', 'ul', 'main', 'article']
        tpo_rel_feature_potential = [col for col in tpo_rel_feature_potential if col not in high_occurrence]
        tpo_irrel_feature_potential = [col for col in tpo_irrel_feature_potential if col not in high_occurrence]
        df = check_columns_on_text(df, 'tree_path', high_occurrence, tpo_rel_feature_potential, tpo_irrel_feature_potential)
        print('Successfully adding overall tree-path feature!\n')
    except Exception as e:
         print("Error:", e)
         
    try:
        # Text-tag analysis
        if train_or_val == 'train':
            _, tt_rel_feature_potential, tt_irrel_feature_potential = column_investigator(df, 'text_tag', 1, 0.5, 2)

        high_occurrence = ['a', 'li', 'figure']
        tt_rel_feature_potential = [col for col in tt_rel_feature_potential if col not in high_occurrence]
        tt_irrel_feature_potential = [col for col in tt_irrel_feature_potential if col not in high_occurrence]
        df = check_columns_on_text(df, 'text_tag', high_occurrence, tt_rel_feature_potential, tt_irrel_feature_potential)
        print('Successfully adding text-tag feature!\n')
    except Exception as e:
         print("Error:", e)
         
    try:
        # Height analysis
        height_bins = [-1000, 150, 400, 1000000]
        height_labels = ['Min - 150', '150 - 400', '400 - Max']
        if train_or_val == 'train':
            _, h_rel_feature_potential, h_irrel_feature_potential = dimension_investigator(df, height_bins, height_labels, 'height', 1, 0.5, 2)

        df = dim_binner(df, 'height', height_bins, height_labels)
        df = check_columns_on_dim(df, 'height_binned', h_rel_feature_potential, h_irrel_feature_potential)
        print('Successfully adding height-based feature!\n')
    except Exception as e:
         print("Error:", e)
         
    try:
        # Sizes analysis
        df['sizes_length'] = df['sizes'].apply(lambda x: len(x.split(',')) if x else None).dropna()
        if train_or_val == 'train':
            _, s_rel_feature_potential, s_irrel_feature_potential = column_investigator(df, 'sizes_length', 0, 0.5, 2)

        df = check_columns_on_dim(df, 'sizes_length', s_rel_feature_potential, s_irrel_feature_potential)
        print('Successfully adding sizes-based feature!\n')
    except Exception as e:
         print("Error:", e)
         
    try:
        # Scrset analysis
        df['srcset_length'] = df['srcset'].apply(lambda x: len(x.split(',')) if x else None).dropna()
        if train_or_val == 'train':
            _, ss_rel_feature_potential, ss_irrel_feature_potential = column_investigator(df, 'srcset_length', 0, 0.5, 2)

        df = check_columns_on_dim(df, 'srcset_length', ss_rel_feature_potential, ss_irrel_feature_potential)
        print('Successfully adding srcset-based feature!\n')
    except Exception as e:
         print("Error:", e)
         
    try:
        # Width analysis
        width_bins = [-1000, 150, 350, 1000000]
        width_labels = ['Min - 150', '150 - 350', '350 - Max']
        if train_or_val == 'train':
            _, w_rel_feature_potential, w_irrel_feature_potential = dimension_investigator(df, width_bins, width_labels, 'width', 0, 0.5, 2)

        df = dim_binner(df, 'width', width_bins, width_labels)
        df = check_columns_on_dim(df, 'width_binned', w_rel_feature_potential, w_irrel_feature_potential)
        print('Successfully adding width-based feature!\n')
    except Exception as e:
         print("Error:", e)
         
    try:
        # Class analysis
        if train_or_val == 'train':
            _, c_rel_feature_potential, c_irrel_feature_potential = column_investigator(df, 'class', 0.1, 0.5, 2)

        df = check_columns_on_text(df, 'class', [], c_rel_feature_potential, c_irrel_feature_potential)
        print('Successfully adding class-based feature!\n')
    except Exception as e:
         print("Error:", e)
         
    try:
        # Style analysis
        if train_or_val == 'train':
            _, st_rel_feature_potential, st_irrel_feature_potential = column_investigator(df, 'style', 0.1, 0.5, 2)

        df = check_columns_on_text(df, 'style', [], st_rel_feature_potential, st_irrel_feature_potential)
        print('Successfully adding style-based feature!\n')
    except Exception as e:
         print("Error:", e)

    try:
        # Drop columns
        columns_to_keep = ['alt_is_null', 
                        #    'sizes_is_null', # overlap with srcset_length
                           'class_is_null',
            'tree_path_first_contains_overlap_a',
            'tree_path_first_contains_overlap_div',
            'tree_path_first_contains_relevant',
            'tree_path_first_contains_irrelevant', 'tree_path_contains_overlap_a',
            # 'tree_path_contains_overlap_li', # overlap with ul
            'tree_path_contains_overlap_ul',
            'tree_path_contains_overlap_main', 'tree_path_contains_overlap_article',
            'tree_path_contains_relevant', 'tree_path_contains_irrelevant',
            'text_tag_contains_overlap_a', 'text_tag_contains_overlap_li',
            'text_tag_contains_overlap_figure', 'text_tag_contains_relevant',
            'text_tag_contains_irrelevant',
            # 'height_binned_contains_relevant', 'height_binned_contains_irrelevant', # overlap with width columns
            'sizes_length_contains_relevant', 'sizes_length_contains_irrelevant',
            'srcset_length_contains_relevant', 'srcset_length_contains_irrelevant','width_binned_contains_relevant',
            'width_binned_contains_irrelevant', 'class_contains_relevant',
            'class_contains_irrelevant', 'style_contains_relevant',
            'style_contains_irrelevant', 'check_title', 'check_alt', 'check_text',
            'check_url_page', 'check_src', 'check_source']
        unused_columns = [column for column in df.columns if column not in columns_to_keep]
        df = column_dropper(df, unused_columns)
        print('Successfully removing high correlated features!\n')
    except Exception as e:
         print("Error:", e)

    try:
        df = df[columns_to_keep]
        df.fillna(0, inplace=True)
        print('Successfully reordering columns!\n')
    except Exception as e:
         print("Error:", e)         

    output = [df, null_check_columns, null_columns_to_drop, 
              tpf_rel_feature_potential,  tpf_irrel_feature_potential, 
              tpo_rel_feature_potential,  tpo_irrel_feature_potential, 
              tt_rel_feature_potential,  tt_irrel_feature_potential, 
              h_rel_feature_potential,  h_irrel_feature_potential, 
              s_rel_feature_potential,  s_irrel_feature_potential, 
              ss_rel_feature_potential,  ss_irrel_feature_potential, 
              w_rel_feature_potential,  w_irrel_feature_potential, 
              c_rel_feature_potential,  c_irrel_feature_potential, 
              st_rel_feature_potential,  st_irrel_feature_potential
              ]

    return output

    
