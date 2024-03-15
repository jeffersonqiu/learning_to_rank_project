import pandas as pd
import matplotlib.pyplot as plt

def null_columns_checker(df, lower_threshold, upper_threshold):
    relevant_df = df[df['is_relevant'] == 1]
    irrelevant_df = df[df['is_relevant'] == 0]
    
    a = round((relevant_df.isnull().sum())/len(relevant_df)*100, 1)
    b = round((irrelevant_df.isnull().sum())/len(irrelevant_df)*100, 1)
    overall = round((relevant_df.isnull().sum() + irrelevant_df.isnull().sum())/(len(df))*100, 1)
    
    missing_values_df = pd.DataFrame({
        'Rel: % of Null': a,
        'Irrel: % of Null': b,
        'Avg': overall,
        'Ratio': round(a/b,2)
    })

    missing_values_df['Null Check Potential'] = missing_values_df['Ratio'].apply(lambda x: True if (x <= lower_threshold or x >= upper_threshold) else False)
    null_check_columns = list(missing_values_df[missing_values_df['Null Check Potential']==True].index)
    columns_to_drop = list(missing_values_df[missing_values_df['Avg']>=99].index)

    missing_values_df.drop('Avg', axis=1, inplace=True)

    print("null_check_columns: ", null_check_columns)
    print("columns_to_drop: ", columns_to_drop)
    
    # Display the combined table
    return missing_values_df, null_check_columns, columns_to_drop

# A function to derive Ratio, i.e. discrepancies in behaviour for relevant and irrelevant group
def column_investigator(df, column_to_investigate, minimum_occurrence, lower_threshold, upper_threshold):
    total_data = len(df)
    relevant_doc_count = sum(df['is_relevant'] == 1) 
    irrelevant_doc_count = total_data - relevant_doc_count
    df_count = df.groupby(column_to_investigate).agg(
        relevant_count=('is_relevant', 'sum'),
        total_count=('is_relevant', 'count')
    )
    df_count.rename({
        'relevant_count': 'Rel Count',
        'total_count': 'Total Count',
    }, inplace=True, axis=1)
    df_count['Irrel Count'] = df_count['Total Count'] - df_count['Rel Count']
    df_count['Occurence in Data'] = round(df_count['Total Count'] / total_data * 100, 1)
    df_count['Rel: % occurrence'] = round((df_count['Rel Count'] / relevant_doc_count) * 100, 1)
    df_count['Irrel: % occurrence'] = round((df_count['Irrel Count'] / irrelevant_doc_count) * 100, 1)
    df_count['Ratio'] = round(df_count['Rel: % occurrence'] / df_count['Irrel: % occurrence'], 2)

    output_df = df_count[df_count['Occurence in Data'] > minimum_occurrence].sort_values(by='Occurence in Data', ascending=False)
    output_df['Rel Feature Potential'] = df_count['Ratio'].apply(lambda x: True if x >= upper_threshold else False)
    output_df['Irrel Feature Potential'] = df_count['Ratio'].apply(lambda x: True if x <= lower_threshold else False)

    rel_feature_potential = list(output_df[output_df['Rel Feature Potential']==True].index)
    irrel_feature_potential = list(output_df[output_df['Irrel Feature Potential']==True].index)

    output_df.drop(['Rel Feature Potential', 'Irrel Feature Potential'], axis=1, inplace=True)

    output_df = output_df[['Rel Count', 'Irrel Count', 'Total Count', 'Occurence in Data',
       'Rel: % occurrence', 'Irrel: % occurrence', 'Ratio']]

    print("rel_feature_potential: ", rel_feature_potential)
    print("irrel_feature_potential: ", irrel_feature_potential)
    combined_features = rel_feature_potential + irrel_feature_potential
    print(f"Total Data covered: {round(sum(output_df.loc[combined_features]['Occurence in Data']), 1)}%")
    
    return output_df, rel_feature_potential, irrel_feature_potential

def find_unique_values(df, col):
    unique_sections = set()
    for path in df[col]:
        # Check if path is iterable; skip if it's None or not an iterable
        if path is not None:
            try:
                unique_sections.update(path)
            except TypeError:  # Catch if 'path' is not iterable
                # Optionally, handle unexpected non-iterable types (e.g., integers, floats, etc.)
                unique_sections.add(path)  # Treat the non-iterable as a single value
    return unique_sections

def analyser_generic(df, text_set, column):
    # Initialize containers for percentage and count
    percentages = {}
    counts = {}
    text_set.discard(None)
    # Analyze by section
    for text in text_set:
        # Determine if each path contains the section
        contains = df[column].apply(lambda x: text in x if x is not None else False)
        # Calculate and store percentage and count
        percentages[text] = round(contains.mean() * 100, 2)  # Convert fraction to percentage
        counts[text] = round(contains.sum())  # Total occurrences

    # Create a DataFrame from the calculated data
    results_df = pd.DataFrame([percentages, counts], index=['percentage', 'total_count'])

    # Transpose the DataFrame to have sections as rows and metrics (Percentage, Total Count) as columns
    transposed_results_df = results_df.T

    return transposed_results_df

def tree_path_investigator(df, text_set, column_to_investigate, minimum_occurrence, lower_threshold, upper_threshold):
    total_data = len(df)
    relevant_df = df[df['is_relevant'] == 1]
    irrelevant_df = df[df['is_relevant'] == 0]
    path_rel_count = analyser_generic(relevant_df, text_set, column_to_investigate)
    path_irrel_count = analyser_generic(irrelevant_df, text_set, column_to_investigate)
    
    path_rel_count_renamed = path_rel_count.rename(columns=lambda x: f'{column_to_investigate}_rel_{x}')
    path_irrel_count_renamed = path_irrel_count.rename(columns=lambda x: f'{column_to_investigate}_irrel_{x}')
    
    path_combined_df = pd.concat([path_rel_count_renamed, path_irrel_count_renamed], axis=1)
    path_combined_df[f'{column_to_investigate}_total_count'] =path_combined_df[f'{column_to_investigate}_rel_total_count'] + path_combined_df[f'{column_to_investigate}_irrel_total_count']
    path_combined_df.sort_values(by=f'{column_to_investigate}_total_count', ascending=False, inplace=True)
    
    # Get the 'total_count' value for 'html'
    html_total_count = path_combined_df.at['html', 'tree_path_total_count']
    
    # Create a new column by dividing all 'total_count' values by the 'html' total_count value
    path_combined_df['tree_path_relative_count'] = path_combined_df['tree_path_total_count'] * 100 / total_data
    path_combined_df['ratio'] = round(path_combined_df['tree_path_rel_percentage'] / path_combined_df['tree_path_irrel_percentage'], 2)
    
    path_combined_df = path_combined_df[['tree_path_rel_total_count', 'tree_path_irrel_total_count', 'tree_path_total_count',
                                        'tree_path_relative_count', 'tree_path_rel_percentage', 'tree_path_irrel_percentage', 'ratio']]
    
    path_combined_df.rename({
        'tree_path_rel_total_count': 'Rel Count',
        'tree_path_total_count': 'Total Count',
        'tree_path_irrel_total_count': 'Irrel Count',
        'tree_path_relative_count': 'Occurence in Data',
        'tree_path_rel_percentage': 'Rel: % occurrence',
        'tree_path_irrel_percentage': 'Irrel: % occurrence',
        'ratio': 'Ratio'
    }, axis=1, inplace=True)

    path_combined_df['Rel Feature Potential'] = path_combined_df['Ratio'].apply(lambda x: True if x >= upper_threshold else False)
    path_combined_df['Irrel Feature Potential'] = path_combined_df['Ratio'].apply(lambda x: True if x <= lower_threshold else False)

    rel_feature_potential = list(path_combined_df[path_combined_df['Rel Feature Potential']==True].index)
    irrel_feature_potential = list(path_combined_df[path_combined_df['Irrel Feature Potential']==True].index)

    path_combined_df.drop(['Rel Feature Potential', 'Irrel Feature Potential'], axis=1, inplace=True)

    print("rel_feature_potential: ", rel_feature_potential)
    print("irrel_feature_potential: ", irrel_feature_potential)
    combined_features = rel_feature_potential + irrel_feature_potential
    print(f"Total Data covered: {round(sum(path_combined_df.loc[combined_features]['Occurence in Data']), 1)}%")

    return path_combined_df, rel_feature_potential, irrel_feature_potential

# To check how many entries consist of non-numeric character
def non_numeric_check(df, column_to_investigate):
    relevant_doc_count = sum(df['is_relevant'] == 1)
    irrelevant_doc_count = sum(df['is_relevant'] == 0)
    
    # Drop rows where 'height' or 'is_relevant' is NaN
    cleaned_data = df.dropna(subset=[column_to_investigate, 'is_relevant'])
    
    # Initialize counters
    str_count_relevant = 0
    str_count_irrelevant = 0
    
    set_count_relevant = set()
    set_count_irrelevant = set()
    
    # Function to check if the value can be converted to a numerical
    def is_convertible_to_int(value):
        try:
            float(value)  # Try converting to numerical
            return True
        except ValueError:  # If conversion fails, it's not a numerical
            return False
    
    # Iterate through the DataFrame and count
    for index, row in cleaned_data.iterrows():
        # Check if the 'height' value is a non-numerical string
        if not is_convertible_to_int(row[column_to_investigate]):
            # Check the 'is_relevant' status and increment the appropriate counter
            if row['is_relevant'] == 1:
                str_count_relevant += 1
                set_count_relevant.add(row[column_to_investigate])
            elif row['is_relevant'] == 0:
                str_count_irrelevant += 1
                set_count_irrelevant.add(row[column_to_investigate])
    
    # Display the results
    print(f'Non-numeric string count when relevant: {round(str_count_relevant/ relevant_doc_count* 100, 1)}%')
    print(f'Non-numeric string count when irrelevant: {round(str_count_irrelevant/ irrelevant_doc_count*100, 1)}%')

# To divide numberical columns to histograms
def dimension_investigator(df, bins, labels, column_to_investigate, minimum_occurrence, lower_threshold, upper_threshold):  
    # Ensure 'height' is properly converted, replacing 'None' with 'NaN'
    df['height'] = pd.to_numeric(df['height'], errors='coerce')
    
    # Now apply the pd.cut with bins and labels
    df['binned'] = pd.cut(df['height'], bins=bins, labels=labels, right=False)
    
    # Since 'pd.cut' here may not immediately cause the issue, the error might be misleading.
    # Add 'Missing' category for NaN values in 'height'
    df['binned'] = df['binned'].astype(pd.CategoricalDtype(categories=labels + ['Missing'], ordered=True))
    df['binned'].fillna('Missing', inplace=True)
    
    # Split the DataFrame based on the 'label'
    relevant_df = df[df['is_relevant'] == 1]
    irrelevant_df = df[df['is_relevant'] == 0]
    
    relevant_doc_count = len(relevant_df)
    irrelevant_doc_count = len(irrelevant_df)
    total_doc = relevant_doc_count + irrelevant_doc_count
    
    # Create the histogram tables for each subset
    hist_table_relevant = relevant_df['binned'].value_counts().sort_index()
    hist_table_not_relevant = irrelevant_df['binned'].value_counts().sort_index()
    
    # Convert to DataFrame for nicer table format
    hist_table_relevant_df = hist_table_relevant.reset_index()
    hist_table_relevant_df.columns = ['Range', 'Frequency (Relevant)']
    hist_table_relevant_df['Frequency (Relevant) %'] = round(hist_table_relevant_df['Frequency (Relevant)'] * 100 / relevant_doc_count, 2)
    
    hist_table_not_relevant_df = hist_table_not_relevant.reset_index()
    hist_table_not_relevant_df.columns = ['Range', 'Frequency (Irrelevant)']
    hist_table_not_relevant_df['Frequency (Irrelevant) %'] = round(hist_table_not_relevant_df['Frequency (Irrelevant)'] * 100 / irrelevant_doc_count, 2)
    
    # Optionally, you can merge these tables for side-by-side comparison
    hist_table_combined_df = pd.merge(hist_table_relevant_df, hist_table_not_relevant_df, on='Range', how='outer').fillna(0)
    
    hist_table_combined_df['Occurence in Data'] = round((hist_table_combined_df['Frequency (Relevant)'] + 
                                                         hist_table_combined_df['Frequency (Irrelevant)']) * 100/ total_doc, 2)
     
    hist_table_combined_df['Ratio'] = hist_table_combined_df['Frequency (Relevant) %'] / hist_table_combined_df['Frequency (Irrelevant) %']
    
    hist_table_combined_df.rename({
        'Frequency (Relevant)': 'Rel Count',
        'Frequency (Irrelevant)': 'Irrel Count',
        'Frequency (Relevant) %': 'Rel: % occurrence',
        'Frequency (Irrelevant) %': 'Irrel: % occurrence',
    }, axis=1, inplace=True)
    
    hist_table_combined_df['Rel Feature Potential'] = hist_table_combined_df['Ratio'].apply(lambda x: True if x >= upper_threshold else False)
    hist_table_combined_df['Irrel Feature Potential'] = hist_table_combined_df['Ratio'].apply(lambda x: True if x <= lower_threshold else False)
    
    rel_feature_potential = list(hist_table_combined_df[hist_table_combined_df['Rel Feature Potential']==True]['Range'])
    irrel_feature_potential = list(hist_table_combined_df[hist_table_combined_df['Irrel Feature Potential']==True]['Range'])
    
    hist_table_combined_df.drop(['Rel Feature Potential', 'Irrel Feature Potential'], axis=1, inplace=True)
    
    print("rel_feature_potential: ", rel_feature_potential)
    print("irrel_feature_potential: ", irrel_feature_potential)
    combined_features = rel_feature_potential + irrel_feature_potential
    print(f"Total Data covered: {round(sum(hist_table_combined_df[hist_table_combined_df['Range'].isin(combined_features)]['Occurence in Data']), 1)}%")
    
    return hist_table_combined_df, rel_feature_potential, irrel_feature_potential

