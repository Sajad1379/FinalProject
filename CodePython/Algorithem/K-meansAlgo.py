from sklearn.cluster import KMeans
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

df=pd.read_excel(r'C:\Users\sajad\Desktop\FinalProject\data.xlsx',engine='openpyxl')

df=df.drop(['final_result'] , axis=1)
df=df.drop(['id_student'] , axis=1)
X = df.drop(['k-means'] , axis=1)

# مقدار تابع هدف برای تعداد مختلفی از خوشه‌ها
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('The Elbow Method')
plt.show()

kmeans = KMeans(n_clusters=5, n_init=26075)

kmeans.fit(X)

data_c=kmeans.labels_

for i in range(0,len(data_c)):
    data.at[i , "LearningModel"]=data_c[i]



data.to_excel(r'C:\Users\sajad\Desktop\FinalProject\data.xlsx', index=False)



