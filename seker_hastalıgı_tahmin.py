#!/usr/bin/env python
# coding: utf-8

# # KNN - K Nearest Neighbours Modeli
# Bu dersimizde Machine Learning modellerinden KNN modelini python'da şeker hastalığı veri setini örneğiyle uygulamalı olarak öğrenceğiz

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split # veri kümesini eğitim ve test alt kümelerine ayırmak için kullanılır
from sklearn.neighbors import KNeighborsClassifier # Bu algoritma, örneklerin birbirlerine olan uzaklıklarını ölçer ve bir örneği, k en yakın komşusunun etiketi ile sınıflandırır.

# Outcome = 1 Diabet/Şeker Hastası
# Outcome = 0 Sağlıklı
data = pd.read_csv("C:\\Users\\pc\\Desktop\\Artificial İntelligence\\PROJELER\\seker_hastalığı_tahmin\\diabetes.csv")
data.head()


# In[10]:


seker_hastalari = data[data.Outcome == 1] # dataframe'i filtreledik yeni bir df oluşturdum.
saglikli_insanlar = data[data.Outcome == 0]

# şimdilik sadece glikoz'a bakarak bir çizim yapalım:
plt.scatter(saglikli_insanlar.Age,saglikli_insanlar.Glucose,color="green",label="sağlıklı",alpha=0.4)
plt.scatter(seker_hastalari.Age,seker_hastalari.Glucose,color="red",label="diyabet",alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()


# In[11]:


# x ve y eksenlerini belirleryelim
y = data.Outcome.values # kişi hasta mı değil mi ? 
x_ham_veri = data.drop(["Outcome"],axis=1) # outcome çıkarılır bağımsız değişkenlere dayalı olarak bir kişinin hastalık durumunu tahmin etmeye çalışan bir sınıflandırma modeli oluşturulabilir.
# Outcome sütununu (dependent variable) çıkarıp sadece independent variables bırakıyoruz.
# çünkü KNN  algoritması x değerleri içerisinde gruplandırma yapacak.

# normalization yapıyoruz - x_ham_veri içerisindeki değerleri sadece 0 ve 1 arasında olacak şekilde hepsini güncelliyoruz.
# eğer bu şekilde normalization yapmazsak yüksek rakamları ezer ve KNN algoritmsı yanıltabilir.
x = (x_ham_veri - np.min(x_ham_veri))/(np.max(x_ham_veri)-np.min(x_ham_veri))

# önce
print("Normalization öncesi ham veriler: \n")
print(x_ham_veri.head())

#sonra
print("\n\n\nNormalization sonrası yapay zekaya eğitim için vereceğimiz veriler: \n")
print(x.head())

# Her özellik değerinden, o özelliğin minimum değerini çıkarırız. Bu, her özellik değerini o özelliğin minimum değerine göre bir ölçekte hareket ettirir.
# Elde edilen değerleri, o özelliğin maksimum ve minimum değerleri arasındaki farka bölerek, 0 ile 1 arasında bir değere dönüştürürüz. Bu, tüm özellik değerlerini aynı aralığa getirir.
# Bu normalizasyon işlemi, özellik değerlerinin farklı ölçeklerde olması durumunda, algoritmaların daha iyi performans göstermesine yardımcı olur. Özellikle K-NN gibi algoritmalar, özellik değerleri arasındaki uzaklıklara dayalı olarak çalıştığından, bu tür bir normalizasyon ölçeklendirme işlemi önemlidir. Bu sayede, bir özellik diğerine göre daha büyük bir ölçekte olursa, bu özellik diğerlerini domine etmez ve algoritma daha dengeli çalışır.


# In[20]:


# train datamız ile test datamızı ayırıyoruz.
# train datamız sistemin sağlıklı insan ile hasta insanı ayırt etmesini öğrenmek için kullanılacak
# test datamız ise bakalım makine öğrenme modelimiz doğru bir şekilde hasta ve sağlıklı insanları ayırt edebiliyor mu diye test etme için kullanılacak
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state=1)
# KNN modelimizi oluşturuyoruz.
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(x_train,y_train) # modeli eğit demek fit
predection = knn.predict(x_test)
print("K = 3 için test verilerimizin doğrulama testi sonucu ",knn.score(x_test,y_test)) # score fonksiyonu, modelin doğruluk oranını hesaplamak için kullanılır. 

# sistem  %74 başarılı %26 başarısız çalışıyor.


# In[21]:


# k kaç olmalı ?
# en iyi k değerini belirleyelim...
sayac = 1
for k in range(1,11):
    knn_yeni = KNeighborsClassifier(n_neighbors = k)
    knn_yeni.fit(x_train,y_train)
    print(sayac," ","Doğruluk Oranı: %",knn_yeni.score(x_test,y_test)*100)
    sayac+=1


# In[26]:


# Yeni bir hasta tahmini için:
from sklearn.preprocessing import MinMaxScaler

# normalization yapıyoruz - daha hızlı normalization yapabilmek için MinMax  scaler kullandık...
sc = MinMaxScaler()
sc.fit_transform(x_ham_veri)

new_prediction = knn.predict(sc.transform(np.array([[6,148,72,35,0,33.6,0.627,50]])))
new_prediction[0]


# In[ ]:




