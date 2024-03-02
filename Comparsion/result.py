import pandas as pd
import matplotlib.pyplot as plt

# Create a dictionary with the metric values
data = {
    'Algorithm': ['KNN', 'Naive Bayes', 'SVM', 'Logistic Regression'],
    'Accuracy on Training data': [0.9927, 0.8390, 0.8695, 0.8524],
    'F1 Score': [0.9378, 0.7907, 0.8435, 0.8230],
    'Precision': [0.9423, 0.7727, 0.7760, 0.7686],
    'Recall': [0.9333, 0.8095, 0.9238, 0.8857],
    'Overall Accuracy': [0.9366, 0.7805, 0.8244, 0.8049]
}

# Create a DataFrame
df = pd.DataFrame(data)
print(df)

# Plot a separate bar chart for each metric in individual figures
metrics = ['Accuracy on Training data', 'F1 Score', 'Precision', 'Recall', 'Overall Accuracy']

for metric in metrics:
    plt.figure(figsize=(8, 6))
    plt.bar(df['Algorithm'], df[metric])
    plt.xlabel('Algorithm')
    plt.ylabel(metric)
    plt.title(f'{metric} Comparison')
    plt.show()


data = {
    'Algorithm': ['KNN', 'Naive Bayes', 'SVM', 'Logistic Regression'],
    'Accuracy on Training data': [0.9927, 0.8390, 0.8695, 0.8524],
    'F1 Score': [0.9378, 0.7907, 0.8435, 0.8230],
    'Precision': [0.9423, 0.7727, 0.7760, 0.7686],
    'Recall': [0.9333, 0.8095, 0.9238, 0.8857],
    'Overall Accuracy': [0.9366, 0.7805, 0.8244, 0.8049]
}

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(6, 4))
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df.values, colLabels=df.columns, cellLoc = 'center', loc='center')

plt.savefig('metrics_comparison.png')
plt.show()

