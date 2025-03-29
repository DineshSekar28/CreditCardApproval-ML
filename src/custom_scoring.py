from src.randomFlip import introduce_label_noise

def custom_scoring(df):
    def score_income(income):
        incomeScore = round(income // 100000)
        return min(incomeScore, 5)

    def score_family(familyCount):
        if familyCount <= 4:
            return 2
        else:
            extra_members = familyCount - 4
            negScore = extra_members // 2  # -1 for every 2 extra members
            return 2 - negScore

    # Education level scores
    education_scores = {
        'Academic degree': 4,
        'Higher education': 3,
        'Incomplete higher': 2,
        'Secondary / secondary special': 1,
        'Lower secondary': 0
    }

    #Income type scores
    income_type_scores = {
        'State servant': 4,
        'Working': 3,
        'Commercial associate': 2,
        'Pensioner': 1,
        'Student': 0
    }

    housing_type_scores = {
        'Co-op apartment': 4,
        'House / apartment': 3,
        'Municipal apartment': 2,
        'Rented apartment': 1,
        'With parents': 1,
        'Office apartment': 0
    }

    # Combine into overall credit score
    def calculate_credit_score(row):
        score = 0
        score += score_income(row['AMT_INCOME_TOTAL'])
        score += income_type_scores.get(row['NAME_INCOME_TYPE'], 0)
        score += housing_type_scores.get(row['NAME_HOUSING_TYPE'], 0)
        score += row.get('EDUCATION_SCORE', 0)
        score += score_family(row['CNT_FAM_MEMBERS'])

        if row['WORK_EXPERIENCE'] > 5:
            score += 2
        if 25 <= row['AGE'] <= 55:
            score += 2
        if row['FLAG_OWN_CAR'] == 'Y':
            score += 1
        if row['FLAG_OWN_REALTY'] == 'Y':
            score += 2
        return score

    # Apply scoring logic
    df['CREDIT_SCORE'] = df.apply(calculate_credit_score, axis=1)

    # Set the cutoff at the 40th percentile (top 60% get TARGET = 1)
    score_cutoff = df['CREDIT_SCORE'].quantile(0.4)

    # Assign TARGET based on the cutoff
    df['TARGET'] = df['CREDIT_SCORE'].apply(lambda x: 1 if x >= score_cutoff else 0)

    # Display class distribution
    print("Class Distribution (TARGET):")
    print(df['TARGET'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))

    df =introduce_label_noise(df)

    return df


