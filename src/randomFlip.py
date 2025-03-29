import numpy as np


def introduce_label_noise(df, target_col='TARGET', flip_fraction=0.15, random_state=42):
    np.random.seed(random_state)

    # Get indexes of class 0 and class 1 separately
    idx_zeros = df[df[target_col] == 0].index
    idx_ones = df[df[target_col] == 1].index

    # How many to flip?
    num_flip_zeros = int(len(idx_zeros) * flip_fraction)
    num_flip_ones = int(len(idx_ones) * flip_fraction)

    # Randomly sample indexes to flip
    flip_zeros = np.random.choice(idx_zeros, size=num_flip_zeros, replace=False)
    flip_ones = np.random.choice(idx_ones, size=num_flip_ones, replace=False)

    # Flip them
    df.loc[flip_zeros, target_col] = 1
    df.loc[flip_ones, target_col] = 0

    print(f" Introduced noise: flipped {num_flip_zeros} zeros → 1 and {num_flip_ones} ones → 0.")
    return df