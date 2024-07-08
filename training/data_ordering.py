import pandas as pd
import numpy as np
import os
import inspect


# # Set up the threshold values for Easy/Medium/Hard questions
## MedQA - LLM difficulty thresholds (four models)
# easy_threshold=0.66
# med_threshold=0.80

category_col='cluster_label'
difficulty_metric='llm_difficulty'


# Helper functions
def arrange_emh(group, difficulty_metric, seed, easy_threshold=easy_threshold, med_threshold=med_threshold):
    # Make sure the difficulty_metric is a string representing the column name
    easy = group.query(f"{difficulty_metric} <= {easy_threshold}").sample(frac=1, random_state=seed)
    medium = group.query(f"{easy_threshold} < {difficulty_metric} <= {med_threshold}").sample(frac=1, random_state=seed)
    hard = group.query(f"{difficulty_metric} > {med_threshold}").sample(frac=1, random_state=seed)
    
    return pd.concat([easy, medium, hard])


def interleaved(df, blocked_learning_fn, category_col=category_col):
    # Apply the blocked learning function 
    df_blocked = blocked_learning_fn(df, category_col)

    # Get unique categories in the order they appear in the blocked learning shuffle
    categories_in_order = df_blocked[category_col].drop_duplicates().tolist()

    # Split the data in each category into three equal parts following the data order
    category_splits = {category: np.array_split(df_blocked[df_blocked[category_col] == category], 3) for category in categories_in_order}

    # Interleave these parts from each category
    interleaved_list = []
    for i in range(3):  # Assuming three splits
        for category in categories_in_order:
            if len(category_splits[category]) > i:  # Check if the split exists
                interleaved_list.append(category_splits[category][i])

    # Ensure all elements in interleaved_list are DataFrames
    interleaved_list = [x for x in interleaved_list if isinstance(x, pd.DataFrame)]

    # Combine all the parts into one DataFrame
    interleaved_df = pd.concat(interleaved_list, ignore_index=True)

    return interleaved_df


def interleaved_emh(df, blocked_learning_fn, category_col=category_col, difficulty_metric=difficulty_metric, easy_threshold=easy_threshold, med_threshold=med_threshold):
    df_blocked = blocked_learning_fn(df, category_col)

    # Split the sorted_df by difficulty levels - keep the same order of categories as the original df
    easy_df = df_blocked.query(f"{difficulty_metric} <= {easy_threshold}")
    medium_df = df_blocked.query(f"{easy_threshold} < {difficulty_metric} <= {med_threshold}")
    hard_df = df_blocked.query(f"{difficulty_metric} > {med_threshold}")
    
    # Function to interleave questions within each difficulty level, maintaining category order
    def interleave_questions(df):
        interleaved = pd.DataFrame()
        for category in df[category_col].unique():
            category_questions = df[df[category_col] == category]
            interleaved = pd.concat([interleaved, category_questions])
        return interleaved

    # Interleave questions from each difficulty level
    interleaved_easy = interleave_questions(easy_df)
    interleaved_medium = interleave_questions(medium_df)
    interleaved_hard = interleave_questions(hard_df)
    
    # Concatenate the interleaved questions back into a single DataFrame
    final_df = pd.concat([interleaved_easy, interleaved_medium, interleaved_hard]).reset_index(drop=True)
    
    return final_df



# Data ordering functions
def original(df):
    return df

def random_shuffle_1(df, seed=42):
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

def random_shuffle_2(df, seed=123):
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

def random_shuffle_3(df, seed=456):
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

def random_shuffle_4(df, seed=789):
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)



def curriculum_strict(df, difficulty_metric=difficulty_metric):
    return df.sort_values(by=difficulty_metric, ascending=True).reset_index(drop=True)

# def curriculum_strict_reverse(df, difficulty_metric=difficulty_metric):
#     return df.sort_values(by=difficulty_metric, ascending=False).reset_index(drop=True)

def curriculum_emh_1(df, difficulty_metric=difficulty_metric, seed=42):
    final_df = arrange_emh(df, difficulty_metric, seed)
    return final_df

def curriculum_emh_2(df, difficulty_metric=difficulty_metric, seed=123):
    final_df = arrange_emh(df, difficulty_metric, seed)
    return final_df

def curriculum_emh_3(df, difficulty_metric=difficulty_metric, seed=456):
    final_df = arrange_emh(df, difficulty_metric, seed)
    return final_df

def curriculum_emh_4(df, difficulty_metric=difficulty_metric, seed=789):
    final_df = arrange_emh(df, difficulty_metric, seed)
    return final_df



# The blocked learning ones have random category order and random data order in each category
def blocked_1(df, category_col=category_col, seed=1):
    # Group by category, shuffle within each group
    return df.groupby(category_col, group_keys=False).apply(lambda x: x.sample(frac=1, random_state=seed)).reset_index(drop=True)

def blocked_2(df, category_col=category_col):
    # Count the number of items per category and get the category names in descending order of count
    category_counts = df[category_col].value_counts().index.tolist()
    
    # Reorder the DataFrame based on the sorted category list with the most items first
    ordered_df = pd.concat([df[df[category_col] == cat] for cat in category_counts], ignore_index=True)

    return ordered_df

def blocked_3(df, category_col=category_col, seed=42):
    # Group by category, shuffle within each group
    grouped_df = df.groupby(category_col, group_keys=False).apply(lambda x: x.sample(frac=1, random_state=seed)).reset_index(drop=True)
    
    # Extract unique categories, shuffle the list of categories
    unique_categories = grouped_df[category_col].unique()
    np.random.seed(seed) 
    np.random.shuffle(unique_categories)

    # Concatenate the DataFrames for each category in the order of the shuffled list
    shuffled_df = pd.concat([grouped_df[grouped_df[category_col] == cat] for cat in unique_categories], ignore_index=True)
    
    return shuffled_df

def blocked_4(df, category_col=category_col, seed=123):
    # Group by category, shuffle within each group
    grouped_df = df.groupby(category_col, group_keys=False).apply(lambda x: x.sample(frac=1, random_state=seed)).reset_index(drop=True)
    
    # Extract unique categories, shuffle the list of categories
    unique_categories = grouped_df[category_col].unique()
    np.random.seed(seed) 
    np.random.shuffle(unique_categories)

    # Concatenate the DataFrames for each category in the order of the shuffled list
    shuffled_df = pd.concat([grouped_df[grouped_df[category_col] == cat] for cat in unique_categories], ignore_index=True)
    
    return shuffled_df


def blocked_5(df, category_col=category_col, seed=456):
    # Group by category, shuffle within each group
    grouped_df = df.groupby(category_col, group_keys=False).apply(lambda x: x.sample(frac=1, random_state=seed)).reset_index(drop=True)
    
    # Extract unique categories, shuffle the list of categories
    unique_categories = grouped_df[category_col].unique()
    np.random.seed(seed) 
    np.random.shuffle(unique_categories)

    # Concatenate the DataFrames for each category in the order of the shuffled list
    shuffled_df = pd.concat([grouped_df[grouped_df[category_col] == cat] for cat in unique_categories], ignore_index=True)
    
    return shuffled_df



def blocked_curriculum_strict_1(df, category_col=category_col, difficulty_metric=difficulty_metric):
    # Within each category, sort questions by ascending difficulty
    return df.groupby(category_col, group_keys=False).apply(lambda x: x.sort_values(by=difficulty_metric, ascending=True)).reset_index(drop=True)

def blocked_curriculum_strict_2(df, category_col=category_col, difficulty_metric=difficulty_metric):
    # Count the number of items per category and get the category names in descending order of count
    category_counts = df[category_col].value_counts().index.tolist()
    
    # Reorder the DataFrame: sort within each category by difficulty, then concatenate based on category item count
    ordered_df = pd.concat([df[df[category_col] == cat].sort_values(by=difficulty_metric, ascending=True) for cat in category_counts], ignore_index=True)

    return ordered_df

def blocked_curriculum_strict_3(df, category_col=category_col, difficulty_metric=difficulty_metric, seed=42):
    # Group by category, shuffle within each group, then sort by 'difficulty_metric' in descending order
    grouped_df = df.groupby(category_col, group_keys=False).apply(
        lambda x: x.sort_values(by=difficulty_metric, ascending=True)
    ).reset_index(drop=True)
    
    # Extract unique categories, shuffle the list of categories
    unique_categories = grouped_df[category_col].unique()
    np.random.seed(seed) 
    np.random.shuffle(unique_categories)

    # Concatenate the DataFrames for each category in the order of the shuffled list
    shuffled_df = pd.concat([grouped_df[grouped_df[category_col] == cat] for cat in unique_categories], ignore_index=True)
    
    return shuffled_df

def blocked_curriculum_strict_4(df, category_col=category_col, difficulty_metric=difficulty_metric, seed=123):
    # Group by category, shuffle within each group, then sort by 'difficulty_metric' in descending order
    grouped_df = df.groupby(category_col, group_keys=False).apply(
        lambda x: x.sort_values(by=difficulty_metric, ascending=True)
    ).reset_index(drop=True)
    
    # Extract unique categories, shuffle the list of categories
    unique_categories = grouped_df[category_col].unique()
    np.random.seed(seed) 
    np.random.shuffle(unique_categories)

    # Concatenate the DataFrames for each category in the order of the shuffled list
    shuffled_df = pd.concat([grouped_df[grouped_df[category_col] == cat] for cat in unique_categories], ignore_index=True)
    
    return shuffled_df

def blocked_curriculum_strict_5(df, category_col=category_col, difficulty_metric=difficulty_metric, seed=456):
    # Group by category, shuffle within each group, then sort by 'difficulty_metric' in descending order
    grouped_df = df.groupby(category_col, group_keys=False).apply(
        lambda x: x.sort_values(by=difficulty_metric, ascending=True)
    ).reset_index(drop=True)
    
    # Extract unique categories, shuffle the list of categories
    unique_categories = grouped_df[category_col].unique()
    np.random.seed(seed) 
    np.random.shuffle(unique_categories)

    # Concatenate the DataFrames for each category in the order of the shuffled list
    shuffled_df = pd.concat([grouped_df[grouped_df[category_col] == cat] for cat in unique_categories], ignore_index=True)
    
    return shuffled_df



def blocked_emh_1(df, category_col=category_col, difficulty_metric=difficulty_metric, seed=1):
    # Use the arrange_emh function on each group to sort questions into EMH order
    final_df = df.groupby(category_col, group_keys=False).apply(lambda x: arrange_emh(x, difficulty_metric, seed)).reset_index(drop=True)
    return final_df

def blocked_emh_2(df, category_col=category_col, difficulty_metric=difficulty_metric, seed=12):
    # Get categories sorted by their frequency
    category_counts = df[category_col].value_counts().index.tolist()
    
    # Apply arrange_emh for each category and concatenate the results
    ordered_df = pd.concat([arrange_emh(df[df[category_col] == cat], difficulty_metric, seed) for cat in category_counts], ignore_index=True)
    
    return ordered_df

def blocked_emh_3(df, category_col=category_col, difficulty_metric=difficulty_metric, seed=42):
    # Shuffle within each category group first
    grouped_df = df.groupby(category_col, group_keys=False).apply(lambda x: x.sample(frac=1, random_state=seed))

    # Extract unique categories, shuffle the list of categories
    np.random.seed(seed)
    unique_categories = grouped_df[category_col].unique()
    np.random.shuffle(unique_categories)

    # Apply arrange_emh for each shuffled category and concatenate the results
    shuffled_df = pd.concat([arrange_emh(grouped_df[grouped_df[category_col] == cat], difficulty_metric, seed) for cat in unique_categories], ignore_index=True)
    
    return shuffled_df

def blocked_emh_4(df, category_col=category_col, difficulty_metric=difficulty_metric, seed=123):
    # Shuffle within each category group first
    grouped_df = df.groupby(category_col, group_keys=False).apply(lambda x: x.sample(frac=1, random_state=seed))

    # Extract unique categories, shuffle the list of categories
    np.random.seed(seed)
    unique_categories = grouped_df[category_col].unique()
    np.random.shuffle(unique_categories)

    # Apply arrange_emh for each shuffled category and concatenate the results
    shuffled_df = pd.concat([arrange_emh(grouped_df[grouped_df[category_col] == cat], difficulty_metric, seed) for cat in unique_categories], ignore_index=True)
    
    return shuffled_df

def blocked_emh_5(df, category_col=category_col, difficulty_metric=difficulty_metric, seed=456):
    # Shuffle within each category group first
    grouped_df = df.groupby(category_col, group_keys=False).apply(lambda x: x.sample(frac=1, random_state=seed))

    # Extract unique categories, shuffle the list of categories
    np.random.seed(seed)
    unique_categories = grouped_df[category_col].unique()
    np.random.shuffle(unique_categories)

    # Apply arrange_emh for each shuffled category and concatenate the results
    shuffled_df = pd.concat([arrange_emh(grouped_df[grouped_df[category_col] == cat], difficulty_metric, seed) for cat in unique_categories], ignore_index=True)
    
    return shuffled_df




def interleaved_1(df, category_col=category_col):
    # Call the general interleaved function with blocked_1 as the data ordering function
    return interleaved(df, blocked_1, category_col)

def interleaved_2(df, category_col=category_col):
    return interleaved(df, blocked_2, category_col)

def interleaved_3(df, category_col=category_col):
    return interleaved(df, blocked_3, category_col)

def interleaved_4(df, category_col=category_col):
    return interleaved(df, blocked_4, category_col)

def interleaved_5(df, category_col=category_col):
    return interleaved(df, blocked_5, category_col)



def interleaved_curriculum_strict_1(df, category_col=category_col):
    # Call the general interleaved function with blocked_curriculum_strict_1 as the data ordering function
    return interleaved(df, blocked_curriculum_strict_1, category_col)

def interleaved_curriculum_strict_2(df, category_col=category_col):
    return interleaved(df, blocked_curriculum_strict_2, category_col)

def interleaved_curriculum_strict_3(df, category_col=category_col):
    return interleaved(df, blocked_curriculum_strict_3, category_col)

def interleaved_curriculum_strict_4(df, category_col=category_col):
    return interleaved(df, blocked_curriculum_strict_4, category_col)

def interleaved_curriculum_strict_5(df, category_col=category_col):
    return interleaved(df, blocked_curriculum_strict_5, category_col)



def interleaved_emh_1(df, category_col=category_col, difficulty_metric=difficulty_metric):
    # Call the interleaved_emh function with blocked_emh_1 as the data ordering function
    return interleaved_emh(df, blocked_emh_1, category_col, difficulty_metric)

def interleaved_emh_2(df, category_col=category_col, difficulty_metric=difficulty_metric):
    return interleaved_emh(df, blocked_emh_2, category_col, difficulty_metric)

def interleaved_emh_3(df, category_col=category_col, difficulty_metric=difficulty_metric):
    return interleaved_emh(df, blocked_emh_3, category_col, difficulty_metric)

def interleaved_emh_4(df, category_col=category_col, difficulty_metric=difficulty_metric):
    return interleaved_emh(df, blocked_emh_4, category_col, difficulty_metric)

def interleaved_emh_5(df, category_col=category_col, difficulty_metric=difficulty_metric):
    return interleaved_emh(df, blocked_emh_5, category_col, difficulty_metric)



# Process and reorder datasets based on various data ordering functions
def process_datasets(csv_files, dataset_dir_base, processed_dataset_dir_base, data_ordering_functions,
                     category_col=category_col, difficulty_metric=difficulty_metric, 
                     easy_threshold=easy_threshold, med_threshold=med_threshold): 
    
    for csv_file in csv_files:
        dataset_path = os.path.join(dataset_dir_base, csv_file)
        dataset_name = os.path.splitext(csv_file)[0] # + '_hcat'
        df = pd.read_csv(dataset_path)

        for func in data_ordering_functions:
            params = inspect.signature(func).parameters
            kwargs = {
                'df': df
            }
            if 'category_col' in params:
                kwargs['category_col'] = category_col
            if 'difficulty_metric' in params:
                kwargs['difficulty_metric'] = difficulty_metric
            if 'easy_threshold' in params:
                kwargs['easy_threshold'] = easy_threshold
            if 'med_threshold' in params:
                kwargs['med_threshold'] = med_threshold
            
            processed_df = func(**kwargs)

            # Prepare the directory path for the processed CSV
            processed_csv_dir = os.path.join(processed_dataset_dir_base, dataset_name)
            if not os.path.exists(processed_csv_dir):
                os.makedirs(processed_csv_dir, exist_ok=True)

            # Save the processed DataFrame to a new CSV
            processed_csv_path = os.path.join(processed_csv_dir, f"{func.__name__}.csv")
            processed_df.to_csv(processed_csv_path, index=False)
            print(f"Saved {func.__name__}.csv for {dataset_name}")



if __name__ == "__main__":
    dataset_dir_base = "/code/llm-fine-tuning/CLUSTERING_MEDQA_TRAINING_SETS" # "/code/llm-fine-tuning/medqa_train_data"
    processed_dataset_dir_base = "/code/llm-fine-tuning/CLUSTERING_MEDQA_TRAINING_SETS" # CLUSTERING_MEDQA_TRAINING_SETS
    csv_files = [
        "original.csv"
    ]

    # List of data ordering functions
    data_ordering_functions = [
        original, random_shuffle_1, random_shuffle_2, random_shuffle_3, random_shuffle_4,
        curriculum_strict, curriculum_emh_1, curriculum_emh_2, curriculum_emh_3, curriculum_emh_4,
        blocked_1, blocked_2, blocked_3, blocked_4, blocked_5,
        blocked_curriculum_strict_1, blocked_curriculum_strict_2, blocked_curriculum_strict_3, blocked_curriculum_strict_4, blocked_curriculum_strict_5,
        blocked_emh_1, blocked_emh_2, blocked_emh_3, blocked_emh_4, blocked_emh_5, 
        interleaved_1, interleaved_2, interleaved_3, interleaved_4, interleaved_5,
        interleaved_curriculum_strict_1, interleaved_curriculum_strict_2, interleaved_curriculum_strict_3, interleaved_curriculum_strict_4, interleaved_curriculum_strict_5,
        interleaved_emh_1, interleaved_emh_2, interleaved_emh_3, interleaved_emh_4, interleaved_emh_5,
        # curriculum_strict_reverse, curriculum_emh_reverse_1, curriculum_emh_reverse_2 # Reversed difficulty
    ]

    # Call the main processing function
    process_datasets(csv_files, dataset_dir_base, processed_dataset_dir_base, data_ordering_functions)
