import pandas as pd
import numpy as np

############################################################################################################
# DATA ORDERING WITHOUT REPETITIONS
############################################################################################################

# Helper function to arrange each category by easy, medium, hard
def arrange_emh(group, difficulty_metric, seed):
    # Make sure the difficulty_metric is a string representing the column name
    easy = group.query(f"{difficulty_metric} > 0.94").sample(frac=1, random_state=seed)
    medium = group.query(f"0.94 >= {difficulty_metric} > 0.85").sample(frac=1, random_state=seed)
    hard = group.query(f"{difficulty_metric} <= 0.85").sample(frac=1, random_state=seed)
    
    return pd.concat([easy, medium, hard])

def interleaved(df, blocked_learning_fn, category_col='Category'):
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

def interleaved_emh(df, blocked_learning_fn, category_col='Category', difficulty_metric='Difficulty'):
    df_blocked = blocked_learning_fn(df, category_col)

    # Split the sorted_df by difficulty levels - keep the same order of categories as the original df
    easy_df = df_blocked.query(f"{difficulty_metric} > 0.94")
    medium_df = df_blocked.query(f"0.94 >= {difficulty_metric} > 0.85")
    hard_df = df_blocked.query(f"{difficulty_metric} <= 0.85")
    
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

def curriculum_strict(df, difficulty_metric='Difficulty'):
    return df.sort_values(by=difficulty_metric, ascending=False).reset_index(drop=True)

def curriculum_emh_1(df, difficulty_metric='Difficulty', seed=42):
    final_df = arrange_emh(df, difficulty_metric, seed)
    return final_df

def curriculum_emh_2(df, difficulty_metric='Difficulty', seed=123):
    final_df = arrange_emh(df, difficulty_metric, seed)
    return final_df

# The three blocked learning has random category order and random data order in each category
def blocked_1(df, category_col='Category', seed=1):
    # Group by category, shuffle within each group
    return df.groupby(category_col, group_keys=False).apply(lambda x: x.sample(frac=1, random_state=seed)).reset_index(drop=True)

def blocked_2(df, category_col='Category'):
    # Count the number of items per category and get the category names in descending order of count
    category_counts = df[category_col].value_counts().index.tolist()
    
    # Reorder the DataFrame based on the sorted category list with the most items first
    ordered_df = pd.concat([df[df[category_col] == cat] for cat in category_counts], ignore_index=True)

    return ordered_df

def blocked_3(df, category_col='Category', seed=42):
    # Group by category, shuffle within each group
    grouped_df = df.groupby(category_col, group_keys=False).apply(lambda x: x.sample(frac=1, random_state=seed)).reset_index(drop=True)
    
    # Extract unique categories, shuffle the list of categories
    unique_categories = grouped_df[category_col].unique()
    np.random.seed(seed) 
    np.random.shuffle(unique_categories)

    # Concatenate the DataFrames for each category in the order of the shuffled list
    shuffled_df = pd.concat([grouped_df[grouped_df[category_col] == cat] for cat in unique_categories], ignore_index=True)
    
    return shuffled_df

def blocked_curriculum_strict_1(df, category_col='Category', difficulty_metric='Difficulty'):
    # Within each category, sort questions by descending difficulty
    return df.groupby(category_col, group_keys=False).apply(lambda x: x.sort_values(by=difficulty_metric, ascending=False)).reset_index(drop=True)

def blocked_curriculum_strict_2(df, category_col='Category', difficulty_metric='Difficulty'):
    # Count the number of items per category and get the category names in descending order of count
    category_counts = df[category_col].value_counts().index.tolist()
    
    # Reorder the DataFrame: sort within each category by difficulty, then concatenate based on category item count
    ordered_df = pd.concat([df[df[category_col] == cat].sort_values(by=difficulty_metric, ascending=False) for cat in category_counts], ignore_index=True)

    return ordered_df

def blocked_curriculum_strict_3(df, category_col='Category', difficulty_metric='Difficulty', seed=42):
    # Group by category, shuffle within each group, then sort by 'difficulty_metric' in descending order
    grouped_df = df.groupby(category_col, group_keys=False).apply(
        lambda x: x.sort_values(by=difficulty_metric, ascending=False)
    ).reset_index(drop=True)
    
    # Extract unique categories, shuffle the list of categories
    unique_categories = grouped_df[category_col].unique()
    np.random.seed(seed) 
    np.random.shuffle(unique_categories)

    # Concatenate the DataFrames for each category in the order of the shuffled list
    shuffled_df = pd.concat([grouped_df[grouped_df[category_col] == cat] for cat in unique_categories], ignore_index=True)
    
    return shuffled_df

def blocked_emh_1(df, category_col='Category', difficulty_metric='Difficulty', seed=1):
    # Use the arrange_emh function on each group to sort questions into EMH order
    final_df = df.groupby(category_col, group_keys=False).apply(lambda x: arrange_emh(x, difficulty_metric, seed)).reset_index(drop=True)
    return final_df

def blocked_emh_2(df, category_col='Category', difficulty_metric='Difficulty', seed=123):
    # Get categories sorted by their frequency
    category_counts = df[category_col].value_counts().index.tolist()
    
    # Apply arrange_emh for each category and concatenate the results
    ordered_df = pd.concat([arrange_emh(df[df[category_col] == cat], difficulty_metric, seed) for cat in category_counts], ignore_index=True)
    
    return ordered_df

def blocked_emh_3(df, category_col='Category', difficulty_metric='Difficulty', seed=42):
    # Shuffle within each category group first
    grouped_df = df.groupby(category_col, group_keys=False).apply(lambda x: x.sample(frac=1, random_state=seed))

    # Extract unique categories, shuffle the list of categories
    np.random.seed(seed)
    unique_categories = grouped_df[category_col].unique()
    np.random.shuffle(unique_categories)

    # Apply arrange_emh for each shuffled category and concatenate the results
    shuffled_df = pd.concat([arrange_emh(grouped_df[grouped_df[category_col] == cat], difficulty_metric, seed) for cat in unique_categories], ignore_index=True)
    
    return shuffled_df

def interleaved_1(df, category_col='Category'):
    # Call the general interleaved function with blocked_1 as the data ordering function
    return interleaved(df, blocked_1, category_col)

def interleaved_2(df, category_col='Category'):
    return interleaved(df, blocked_2, category_col)

def interleaved_3(df, category_col='Category'):
    return interleaved(df, blocked_3, category_col)

def interleaved_curriculum_strict_1(df, category_col='Category'):
    # Call the general interleaved function with blocked_curriculum_strict_1 as the data ordering function
    return interleaved(df, blocked_curriculum_strict_1, category_col)

def interleaved_curriculum_strict_2(df, category_col='Category'):
    return interleaved(df, blocked_curriculum_strict_2, category_col)

def interleaved_curriculum_strict_3(df, category_col='Category'):
    return interleaved(df, blocked_curriculum_strict_3, category_col)

def interleaved_emh_1(df, category_col='Category', difficulty_metric='Difficulty'):
    # Call the interleaved_emh function with blocked_emh_1 as the data ordering function
    return interleaved_emh(df, blocked_emh_1, category_col, difficulty_metric)

def interleaved_emh_2(df, category_col='Category', difficulty_metric='Difficulty'):
    return interleaved_emh(df, blocked_emh_2, category_col, difficulty_metric)

def interleaved_emh_3(df, category_col='Category', difficulty_metric='Difficulty'):
    return interleaved_emh(df, blocked_emh_3, category_col, difficulty_metric)





############################################################################################################
# DATA ORDERING WITH REPETITIONS
############################################################################################################

# Helper function to repeat data with shuffling, the concatenate
def repeat_and_shuffle_group(group, m=3, seed=1):
    # Ensure m is at least 1 to include the group unshuffled at least once
    if m < 1:
        m = 1

    # Initialize with the original group to keep its order first
    first_group = [group]

    # Shuffle and add the group m-1 times
    for _ in range(m - 1):
        shuffled_group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        first_group.append(shuffled_group)
        # Optionally, increment the seed to get a different shuffle each time
        seed += 1

    # Concatenate the original group with its shuffled versions
    repeated_and_shuffled_group = pd.concat(first_group, ignore_index=True)

    return repeated_and_shuffled_group
 
def arrange_emh_repeated(group, seed, difficulty_metric='Difficulty', m=3):
    np.random.seed(seed)  # Ensure reproducibility
    easy = group.query(f"{difficulty_metric} > 0.94").sample(frac=1, random_state=seed)
    medium = group.query(f"0.94 >= {difficulty_metric} > 0.85").sample(frac=1, random_state=seed)
    hard = group.query(f"{difficulty_metric} <= 0.85").sample(frac=1, random_state=seed)

    # Repeat and shuffle each difficulty level
    easy_repeated = repeat_and_shuffle_group(easy, m=m, seed=seed)
    medium_repeated = repeat_and_shuffle_group(medium, m=m, seed=seed)
    hard_repeated = repeat_and_shuffle_group(hard, m=m, seed=seed)

    # Concatenate all segments: original followed by their repeated & shuffled versions
    final_df = pd.concat([easy_repeated, medium_repeated, hard_repeated])

    return final_df

def interleaved_repeated(df, blocked_learning_fn, category_col='Category'):
    # Apply the blocked learning function
    df_blocked = blocked_learning_fn(df, category_col)

    # Get unique categories in the order they appear in the blocked learning shuffle
    categories_in_order = df_blocked[category_col].drop_duplicates().tolist()

    # Split the data in each category into three equal parts, applying repeat and shuffle
    category_splits = {
        category: [repeat_and_shuffle_group(np.array_split(df_blocked[df_blocked[category_col] == category], 3)[i]) 
                   for i in range(3)]
        for category in categories_in_order
    }

    # Interleave these parts from each category
    interleaved_list = []
    for i in range(3):  # Assuming three splits
        for category in categories_in_order:
            if len(category_splits[category]) > i:  # Ensure the split exists
                interleaved_list.append(category_splits[category][i])

    # Combine all the parts into one DataFrame
    interleaved_df = pd.concat(interleaved_list, ignore_index=True)

    return interleaved_df

def interleaved_emh_repeated(df, blocked_learning_fn, category_col='Category', difficulty_metric='Difficulty'):
    df_blocked = blocked_learning_fn(df, category_col)
    
    # Ensure unique order of categories as in the blocked df
    categories_in_order = df_blocked[category_col].drop_duplicates().tolist()

    # Function to process each difficulty level
    def process_difficulty(df):
        interleaved = pd.DataFrame()
        for category in categories_in_order:
            category_questions = df[df[category_col] == category]
            if not category_questions.empty:
                # Apply repeat and shuffle to each category group within the difficulty
                interleaved_category = repeat_and_shuffle_group(category_questions)
                interleaved = pd.concat([interleaved, interleaved_category], ignore_index=True)
        return interleaved

    # Process each difficulty level
    easy_df = process_difficulty(df_blocked.query(f"{difficulty_metric} > 0.94"))
    medium_df = process_difficulty(df_blocked.query(f"0.94 >= {difficulty_metric} > 0.85"))
    hard_df = process_difficulty(df_blocked.query(f"{difficulty_metric} <= 0.85"))
    
    # Concatenate the processed questions from each difficulty level
    final_df = pd.concat([easy_df, medium_df, hard_df], ignore_index=True)
    
    return final_df



def random_shuffle_1_repeated(df, m=3, seed=42):
    random_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return repeat_and_shuffle_group(random_df, m, seed)

def random_shuffle_2_repeated(df, m=3, seed=123):
    random_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return repeat_and_shuffle_group(random_df, m, seed)

def curriculum_emh_1_repeated(df, difficulty_metric='Difficulty', seed=42, m=3):
    final_df = arrange_emh_repeated(df, seed, difficulty_metric, m) 
    return final_df

def curriculum_emh_2_repeated(df, difficulty_metric='Difficulty', seed=123, m=3):
    final_df = arrange_emh_repeated(df, seed, difficulty_metric, m)
    return final_df

def blocked_1_repeated(df, category_col='Category'):
    # Group by category, shuffle within each group, then repeat and shuffle
    return df.groupby(category_col, group_keys=False).apply(repeat_and_shuffle_group).reset_index(drop=True)

def blocked_2_repeated(df, category_col='Category'):
    # Get the category names in descending order of count
    category_counts = df[category_col].value_counts().index.tolist()

    # Apply repeat and shuffle to each category, maintaining overall category order
    ordered_df = pd.concat([repeat_and_shuffle_group(df[df[category_col] == cat]) for cat in category_counts], ignore_index=True)

    return ordered_df

def blocked_3_repeated(df, category_col='Category', seed=42, m=3):
    # Group by category, shuffle within each group, then repeat and shuffle
    grouped_df = df.groupby(category_col, group_keys=False).apply(repeat_and_shuffle_group).reset_index(drop=True)
    
    # Shuffle category order
    np.random.seed(seed)
    unique_categories = grouped_df[category_col].unique()
    np.random.shuffle(unique_categories)
    
    # Concatenate in the shuffled category order
    shuffled_df = pd.concat([grouped_df[grouped_df[category_col] == cat] for cat in unique_categories], ignore_index=True)

    return shuffled_df

def blocked_emh_1_repeated(df, category_col='Category', seed=1):
    # Use the arrange_emh function on each group to sort questions into EMH order
    final_df = df.groupby(category_col, group_keys=False).apply(lambda x: arrange_emh_repeated(x, seed)).reset_index(drop=True)
    return final_df

def blocked_emh_2_repeated(df, category_col='Category', seed=123):
    # Get categories sorted by their frequency
    category_counts = df[category_col].value_counts().index.tolist()
    
    # Apply arrange_emh for each category and concatenate the results
    ordered_df = pd.concat([arrange_emh_repeated(df[df[category_col] == cat], seed) for cat in category_counts], ignore_index=True)
    
    return ordered_df

def blocked_emh_3_repeated(df, category_col='Category', seed=42):
    # Shuffle within each category group first
    grouped_df = df.groupby(category_col, group_keys=False).apply(lambda x: x.sample(frac=1, random_state=seed))

    # Extract unique categories, shuffle the list of categories
    np.random.seed(seed)
    unique_categories = grouped_df[category_col].unique()
    np.random.shuffle(unique_categories)

    # Apply arrange_emh for each shuffled category and concatenate the results
    shuffled_df = pd.concat([arrange_emh_repeated(grouped_df[grouped_df[category_col] == cat], seed) for cat in unique_categories], ignore_index=True)
    
    return shuffled_df

def interleaved_1_repeated(df):
    return interleaved_repeated(df, blocked_1)

def interleaved_2_repeated(df):
    return interleaved_repeated(df, blocked_2)

def interleaved_3_repeated(df):
    return interleaved_repeated(df, blocked_3)

def interleaved_emh_1_repeated(df):
    return interleaved_emh_repeated(df, blocked_1)

def interleaved_emh_2_repeated(df):
    return interleaved_emh_repeated(df, blocked_2)

def interleaved_emh_3_repeated(df):
    return interleaved_emh_repeated(df, blocked_3)
