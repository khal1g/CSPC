import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("--- SESSION 1 ---")

df = pd.read_csv("heart.csv")
print("Dataset shape:", df.shape)
print(df.head())

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.hist(df['age'])
plt.title('age')

plt.subplot(2, 2, 2)
plt.hist(df['chol'])
plt.title('chol')

plt.subplot(2, 2, 3)
plt.hist(df['trestbps'])
plt.title('trestbps')

plt.subplot(2, 2, 4)
plt.hist(df['thalach'])
plt.title('thalach')

plt.tight_layout()
plt.savefig("distributions.png")
plt.show()

for col in ['age', 'chol', 'trestbps', 'thalach']:
    res = stats.shapiro(df[col])
    print(col, "p-value:", res.pvalue)
    if res.pvalue > 0.05:
        print(col, "looks normal")
    else:
        print(col, "is NOT normal")


print("\n--- SESSION 2 ---")

group_sick = df[df['target'] == 1]['thalach']
group_healthy = df[df['target'] == 0]['thalach']

ttest_res = stats.mannwhitneyu(group_sick, group_healthy)
print("Mann-Whitney test p-value:", ttest_res.pvalue)

mean_sick = group_sick.mean()
mean_healthy = group_healthy.mean()

sem_sick = group_sick.std() / np.sqrt(len(group_sick))
sem_healthy = group_healthy.std() / np.sqrt(len(group_healthy))

print("Sick group thalach:", mean_sick, "+-", sem_sick)
print("Healthy group thalach:", mean_healthy, "+-", sem_healthy)

plt.figure()
plt.errorbar(['Healthy (0)', 'Sick (1)'], [mean_healthy, mean_sick], yerr=[sem_healthy*2, sem_sick*2], fmt='o')
plt.title("Comparison of thalach")
plt.ylabel("thalach")
plt.savefig("thalach_comparison.png")
plt.show()

corr, pval = stats.pearsonr(df['age'], df['thalach'])
print("Correlation age vs thalach:", corr, "p-value:", pval)

plt.figure()
plt.scatter(df['age'], df['thalach'])
plt.xlabel("age")
plt.ylabel("thalach")
plt.title("Age vs Thalach")
plt.savefig("age_vs_thalach.png")
plt.show()


print("\n--- DATASET 2 ---")

df2 = pd.read_csv("chemicals_cancer.csv")

corr_b, _ = stats.pearsonr(df2['benzene'], df2['malignancy'])
corr_c, _ = stats.pearsonr(df2['cadmium'], df2['malignancy'])

print("Benzene vs Malignancy corr:", corr_b)
print("Cadmium vs Malignancy corr:", corr_c)

filtered_df = df2[(df2['pollution_index'] > 40) & (df2['pollution_index'] < 60)]

corr_b_filt, _ = stats.pearsonr(filtered_df['benzene'], filtered_df['malignancy'])
corr_c_filt, _ = stats.pearsonr(filtered_df['cadmium'], filtered_df['malignancy'])

print("AFTER FILTERING:")
print("Benzene vs Malignancy corr:", corr_b_filt)
print("Cadmium vs Malignancy corr:", corr_c_filt)


print("\n--- BONUS ---")
counts = df['target'].value_counts(normalize=True)
p0 = counts[0]
p1 = counts[1]

entropy = - (p0 * np.log2(p0) + p1 * np.log2(p1))
print("Target entropy:", entropy)