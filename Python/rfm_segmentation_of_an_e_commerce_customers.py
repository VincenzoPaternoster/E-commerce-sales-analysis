# -*- coding: utf-8 -*-
'''
# RFM Segmentation of an e-commerce customers
'''

# Library
!pip install df2tables

## Basic
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sbn
import datetime as dt
from collections import Counter

## Scipy
from scipy import stats
from scipy.spatial.distance import cdist

## Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

## Table
from df2tables import render_nb,render

## Global setting
SEED=30
palette=sbn.color_palette("hls", 5)
plt.rcParams["font.family"] = "monospace"

### Save plot
import os
folder = "plots"
os.makedirs(folder, exist_ok=True)
def save(name):
    plt.savefig(f"{folder}/{name}.png", dpi=300, bbox_inches="tight")


## Dataset
### Upload dataset
data=pd.read_csv("https://raw.githubusercontent.com/VincenzoPaternoster/E-commerce-sales-analysis/refs/heads/main/Data/Edited%20data/fact_sales.txt")

## Show dataset
data

# Count how many orders were cancelled
Counter(data["status_id"]) #### 1117 orders

# Delete cancelled orders
data=data[data["status_id"]!=2] #### 2 is ID of cancelled orders

data # from 9576 rows to 8459 rows

## Get the original date through date_id

## Upload date dateaset
date_data=pd.read_csv("https://raw.githubusercontent.com/VincenzoPaternoster/E-commerce-sales-analysis/refs/heads/main/Data/Edited%20data/dim_date.txt")

## Date Dataset
date_data

"""## 1. Data preprocessing

### 1.1 Merge dataset
"""

# Get the original date through merge procedure

merge_data=data.merge(date_data[["date_id","date_og"]],how="left",on="date_id")

## Convert column date_og in datetime
merge_data["date_og"]=pd.to_datetime(merge_data["date_og"])

## Create variable to make difference between date
diff_date=max(merge_data["date_og"])

merge_data

## Create subset by Customer_id with aggregations of count order_id,sum of amount and days from the last purchase

df=merge_data.groupby(by="customer_id").agg(frequency_order=("order_id","count"),
                                                        recency_day=("date_og", lambda x: (diff_date-max(x)).days),
                                                        monetary_amount=("amount","sum")).reset_index()

df

## Chech for wrong values

## Frequency
print(df[df["frequency_order"]<=0])

## Recency
print(df[df["recency_day"]<0])

## Monetary
print(df[df["monetary_amount"]<=0])

# Are there null values?
df.info()

"""### 1.2 Test skeweness"""

# Test skewness

## Set dimension of charts
fig, axs = plt.subplots(3,1, figsize=(10,8))

## First chart: Frequency distribution
sbn.kdeplot(df["frequency_order"], fill=True, ax=axs[0],color="red")
axs[0].set_title("Frequency distribution")
axs[0].text(x=0.95, y=0.95, s=f"Skewness= {df['frequency_order'].skew().round(2)}", transform=axs[0].transAxes, ha='right', va='top',bbox=dict(boxstyle="round4",fc="w"))

## Second chart: Recency distributiom
sbn.kdeplot(df["recency_day"], fill=True, ax=axs[1],color="orange")
axs[1].set_title("Recency distribution")
axs[1].text(x=0.95, y=0.95, s=f"Skewness= {df['recency_day'].skew().round(2)}", transform=axs[1].transAxes, ha='right', va='top',bbox=dict(boxstyle="round4",fc="w"))

## Third chart: Monetary distribution
sbn.kdeplot(df["monetary_amount"], fill=True, ax=axs[2],color="purple")
axs[2].set_title("Monetary distribution")
axs[2].text(x=0.95, y=0.95, s=f"Skewness= {df['monetary_amount'].skew().round(2)}", transform=axs[2].transAxes, ha='right', va='top',bbox=dict(boxstyle="round4",fc="w"))


plt.tight_layout()
save("RFM_Features_distributions")
plt.show()

### 1.3 Check for outliers


## Are there outliers?

## Frequency
fig,axs=plt.subplots(1,3,figsize=(10,8))
sbn.boxplot(data=df["frequency_order"],ax=axs[0],color="red")
axs[0].set_title("Frequency distribution")

## Recency
sbn.boxplot(data=df["recency_day"],ax=axs[1],color="orange")
axs[1].set_title("Recency distribution") ## The recency boxplot starts at 0, so previously kedplot presented a biased recency distribution.

## Monetary
sbn.boxplot(data=df["monetary_amount"],ax=axs[2],color="purple")
axs[2].set_title("Monetary distribution")

plt.tight_layout()
save("BoxPlot_Outliers")
plt.show()

## Identify outliers
print(np.quantile(df["frequency_order"],0.5))
print(np.quantile(df["monetary_amount"],0.5))
print(np.quantile(df["recency_day"],0.5))
df[(df["frequency_order"]>np.quantile(df["frequency_order"],0.95)) & (df["monetary_amount"]>np.quantile(df["monetary_amount"],0.95))]

'''
## 2. Clustering

### 2.1 Clustering with raw data

#### 2.1.1 How many clusters do we need?
'''

### Define function to choose number of clusters to use

def how_many_clusters(X,title=False):

    # Get intertias and silhouettes
    inertias = []
    silhouettes = []
    clust_diff={}

    for k in range(2, 12):
        kmeans = KMeans(n_clusters=k, random_state=SEED).fit(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, kmeans.labels_))
        clust_diff[k] = sum(np.min(cdist(X, kmeans.cluster_centers_,
                                   'euclidean'), axis=1)) / X.shape[0]


    # Elbow method
    plt.plot(range(2, 12), inertias, marker='^')
    plt.xlabel("k")
    plt.ylabel("Inertia")

    if title:

      plt.title("Elbow Method with raw data")
      save("Elbowmethod_raw data")
    else:
      plt.title("Elbow Method with scaled/transformed data")
      save("Elbowmethod_transformed_data")
    plt.show()

    # SSE score
    print("\nINERTIA (SSE) VALUES:")
    for numb, val in clust_diff.items():
      print(f"{numb} : {val}")
    print("\n")

    # Silhouette score
    plt.plot(range(2, 12), silhouettes, marker='o', color='green')
    plt.xlabel("k")
    plt.ylabel("Score")

    if title:

      plt.title("Silhouette Score with raw data")
      save("Silhouettesocre_raw_data")
    else:
      plt.title("Silhouette Score with scaled/transformed data")
      save("Silhouettesocre_transformed_data")
    plt.show()

## Raw data clusters
how_many_clusters(df[["frequency_order","recency_day","monetary_amount"]],title=True)

"""
#### 2.1.2 Show clusters
"""

# View clusters of model

# Function for view clusters

def view_clust(X,k,title=False):

    # X features
    # k number of clusters

    # KMeans
    kmeans=KMeans(n_clusters=k,init="k-means++",random_state=SEED).fit(X)
    centers=kmeans.cluster_centers_
    y_kmeans=kmeans.predict(X)

    # View clusters

    # 2 FEATURES
    if X.shape[1]==2:

      # Set labels of axes
      plt.xlabel(X.columns[0])
      plt.ylabel(X.columns[1])

      # Plot scatter
      sbn.scatterplot(x=X[X.columns[0]],y=X[X.columns[1]],hue=pd.Categorical(y_kmeans),s=100)
      plt.scatter(centers[:,0],centers[:,1],c="black",s=200,alpha=0.8,marker="*")

      # Write SSD value on chart
      plt.text(1.2,0,f"SSD: {kmeans.inertia_:.2f}",ha='right', va='bottom', transform=plt.gca().transAxes)

      # Set title of chart
      if title:

        plt.title(f"Scatterplot of {X.columns[0]} and {X.columns[1]} with raw data")
        save(f"Clusters_raw2")
      else:

        plt.title(f"Scatterplot of {X.columns[0]} and {X.columns[1]} with scaled/transformed data")
        save(f"Clusters_trans2")
      plt.legend(title="Cluster")

      plt.show()

    # 3 FEATURES
    elif X.shape[1] == 3:

        fig = plt.figure(figsize=(18, 8))

        views = [(30, 45),(10, -60)]  # 3D Views: I decided to use three graphs for the same data because
                                       #           I noticed that only some clusters were visible, so I decided
                                       #           to plot two graphs with two different perspectives to
                                       #           improve the interpretation. ,

        # Get limits for each axes
        xmin, xmax = X[X.columns[0]].min(), X[X.columns[0]].max()
        ymin, ymax = X[X.columns[1]].min(), X[X.columns[1]].max()
        zmin, zmax = X[X.columns[2]].min(), X[X.columns[2]].max()

        # Use a loop to plot two different views of the same graph

        for i, (elev, azim) in enumerate(views, start=1): ## elev= moves the angle from the Y to the Z
                                                          ## azim= moves the angle from the X to the Y

            ax = fig.add_subplot(1, len(views), i, projection='3d') ## Set 1 row, and lenght of views columns and the number of charts
                                                           ## to display two charts side by side

            ax.view_init(elev=elev, azim=azim) # set elevation and azim in degrees instead of radians
                                               # makes it easier to move the angle
            # Set name of axes
            ax.set_xlabel(X.columns[0])
            ax.set_ylabel(X.columns[1])
            ax.set_zlabel(X.columns[2])

            # Assign labels for each cluster
            for cluster in np.unique(y_kmeans): ## For each unique cluster of y_kmeans

                labs = y_kmeans == cluster ## create boolean mask to understand which observations belong to the current cluster

                ax.scatter3D(X.loc[labs, X.columns[0]], # to filter for each column or feature the observations associated to each cluster
                             X.loc[labs, X.columns[1]],
                             X.loc[labs, X.columns[2]],
                             label=f"Cluster {cluster}", s=60, alpha=0.7)

            # Show centroids
            ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2],
                         c='black', s=200, alpha=0.8, marker="*")

            # Force equal limits
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)



            # Show SSD value
            ax.text2D(0.05, 0.95, f"SSE={kmeans.inertia_:.2f}", transform=ax.transAxes)

            # Set title
            title_text = f"Scatterplot of {X.columns[0]}, {X.columns[1]}, {X.columns[2]}"

            if title:
                title_text += f" with raw data ({k} clusters)"
                save(f"Clusters_raw3")

            else:
                title_text += f" with transformed data ({k} clusters)"
                save(f"Cluster_trans3")
            ax.set_title(title_text)

            ax.legend(title="Cluster", loc='upper right')

        plt.tight_layout()

        plt.show()

    else:
      print("Unable to display chart with this number of dimensions")

## Clusters with raw data

view_clust(df["frequency_order","recency_day","monetary_amount"],5,True)

## Clusters with raw data

view_clust(df[["frequency_order","recency_day","monetary_amount"]],6,True)

"""
### 2.2 Clustering with scaled/transformed data
"""

# Apply transformation and standardize data

## Choose transformation and see differences between technique
### Defining function
def analyze_skewness(x):

    # Check if all values in the column are strictly positive
    # This is necessary for log and Box-Cox transformations
    if (df[x] > 0).all():
      fig, ax = plt.subplots(2, 2, figsize=(6,6))
      fig.suptitle(f"{x} skewness")

    #Raw data
      sbn.kdeplot(df[x], fill=True, ax=ax[0,0])
      ax[0,0].set_title("Raw data")

    #Log transformation
      sbn.kdeplot(np.log(df[x]), fill=True, ax=ax[0,1])
      ax[0,1].set_title("Log transformation")

    #Boxcox transformation
      sbn.kdeplot(stats.boxcox(df[x])[0], fill=True, ax=ax[1,0])
      ax[1,0].set_title("Boxcox transformation")

    #Yeo-Johnson
      sbn.kdeplot(stats.yeojohnson(df[x])[0], fill=True, ax=ax[1,1],warn_singular=False)
      ax[1,1].set_title("Yeo-Johnson transformation")

      ## Compare skewness values
      print('Log Transform : The skew coefficient of', df[x].skew().round(2), 'to', np.log(df[x]).skew().round(2))
      print('Box-Cox Transform : The skew coefficient of', df[x].skew().round(2), 'to', pd.Series(stats.boxcox(df[x])[0]).skew().round(2))
      print('Yeo-Johnson Transform : The skew coefficient of', df[x].skew().round(2), 'to', pd.Series(stats.yeojohnson(df[x])[0]).skew().round(2))

    else:
      # If not all values are positive, only Yeo-Johnson is generally applicable without issues
      fig, ax = plt.subplots(1, 2, figsize=(10,4))
      fig.suptitle(f"{x} skewness")

    #Raw data
      sbn.kdeplot(df[x], fill=True, ax=ax[0])
      ax[0].set_title("Raw data")
    #Yeo-Johnson
      sbn.kdeplot(stats.yeojohnson(df[x])[0], fill=True, ax=ax[1],warn_singular=False)
      ax[1].set_title("Yeo-Johnson transformation")

    ## Compare skewness values
      print('Yeo-Johnson Transform : The skew coefficient of', df[x].skew().round(2), 'to', pd.Series(stats.yeojohnson(df[x])[0]).skew().round(2))


    plt.tight_layout()
    save(f"TransformedDistributions {df[x][0]}")
    plt.show()

# Apply transformation and standardize data

## Choose transformation and see differences between technique
### Frequency
analyze_skewness(df.columns[1])

# Apply transformation and standardize data

## Choose transformation and see differences between technique
### Recency
analyze_skewness(df.columns[2])

# Apply transformation and standardize data

## Choose transformation and see differences between technique
### Monetary
analyze_skewness(df.columns[3])

"""
#### 2.1.1 Transforming data
"""

## Transform data with Yeo-Johnson technique
df_yj=pd.DataFrame()
df_yj["Frequency"]=stats.yeojohnson(df[df.columns[1]])[0]
df_yj["Recency"]=stats.yeojohnson(df[df.columns[2]])[0]
df_yj["Monetary"]=stats.yeojohnson(df[df.columns[3]])[0]

print(df[df.columns[[1,2,3]]].head(10))
print(df_yj.head(10))

# Standardisation

## Initialize StandardScaler()
strd=StandardScaler()

## Fit
strd.fit(df_yj)

## Transform
df_std=strd.transform(df_yj)

## Std dataframe
df_std=pd.DataFrame(df_std,columns=["Frequency","Recency","Monetary"])

df_std

"""#### 2.1.2 How many clusters do we need?"""

how_many_clusters(df_std,False)

"""#### 2.1.3 Show clusters"""

## Clustering with transformed data and 5 clusters
view_clust(df_std, 5, False)

# Create plot to compare two clustering
# fig,ax=subplots(2,2,figsize=(14,8))
## Clustering with transformed data and 5 clusters
view_clust(df_std, 5, False)

## Clustering with raw data
view_clust(df[["frequency_order","recency_day","monetary_amount"]],5,True)


"""
### 3.1 How is each cluster represented by RFM features?
"""

## Fit model
mod=KMeans(n_clusters=5,init="k-means++",random_state=SEED).fit(df_std)
df["Cluster"]=mod.predict(df_std)

## Create dataframe with cluster for each customer_id
df_rfm=pd.DataFrame(df_std)
df_rfm.insert(0,"ID",df['customer_id'])
df_rfm["Cluster"]=mod.labels_

df_rfm

## Get long form of df_frm in order to use snake plot

df_melt=pd.melt(frame=df_rfm.reset_index(),
                id_vars=["ID","Cluster"],
                value_vars=["Frequency","Recency","Monetary"],
                var_name="RFM",
                value_name="Value")

df_melt

## Create snake plot
plt.figure(figsize=(8,6))
sbn.lineplot(data=df_melt,x="RFM",y="Value",hue="Cluster",palette=palette)
plt.legend(title="Cluster",loc="upper right")
save("Snakeplot")
plt.show()


"""
### 3.2 What is the percentage of customers for each cluster?
"""

## Create dataframe with percent for each cluster
df_perc=df_melt.groupby("Cluster").agg(Size=("ID","count"))
df_perc.insert(1,"Percent", ((df_perc["Size"]/df_perc["Size"].sum())*100).round(1))

## Reset index in order to get correct columns and index
df_perc=df_perc.reset_index()

df_perc

### Rename columns
df.columns=["ID","Frequency","Recency","Monetary","Cluster"]

### Add Cluster label column
clust_lab = {0: "Lost high-value",
                  1: "Recent low-value",
                  2: "Medium-value inactive",
                  3: "Recent high-value",
                  4: "Inactive low-value"
}

df["Cluster_Label"]=df["Cluster"].map(clust_lab)
df_perc["Cluster_Label"]=df_perc["Cluster"].map(clust_lab)

df["Monetary"]=df["Monetary"].round(2)


## Create pie chart
explode = (0.05, 0.05, 0.05, 0.05, 0.05)
fig, ax = plt.subplots(figsize=(8,6))
ax.pie(df_perc["Percent"], explode=explode,labels=df_perc['Cluster_Label'],autopct='%1.1f%%',
       shadow=True, startangle=90,colors=palette)
ax.legend(title="Cluster",labels=df_perc["Cluster"],loc="upper right",bbox_to_anchor=(1.2,1.1))
save("PieChart")
plt.show()

"""
### 3.3 How relevant is each RFM feature for each cluster?
"""

## Create heatmap with relative percent that compared cluster and population values of RFM

## Get mean for each clusters by RFM features
cluster_avg=df.groupby("Cluster").agg({
    "Frequency": "mean",
    "Recency": "mean",
    "Monetary": "mean"}).round(2).reset_index()

## Get mean of RFM features by original sample in order to compare with cluster_avg
sample_avg=df[["Frequency","Recency","Monetary"]].mean()

cluster_avg

sample_avg

## Calculate the ratio between the average RFM in the clusters and the average RFM in the original sample
relative_rfm=((cluster_avg/sample_avg)-1).drop("Cluster",axis=1)
relative_rfm.index.name="Cluster"
relative_rfm=relative_rfm.rename(columns={"frequency_order":"Frequency","recency_day":"Recency","monetary_amount":"Monetary"})

## Re-order columns
relative_rfm = relative_rfm[["Frequency","Recency","Monetary", ]]

relative_rfm

## Heatmap of ratio RFM for each clusters
sbn.heatmap(data=relative_rfm,annot=True,linewidths=0.5)
save("Heatmap")
plt.show()

"""
## 4. Conclusion
"""

## Compare snake plot (transformed data) and heatmap (original data)
fig,(ax,ax2,ax3)=plt.subplots(1,3,figsize=(18,6))

## Function for border
def add_border(axis, lw=1.5):
    for line in axis.spines.values():
        line.set_visible(True)
        line.set_linewidth(lw)


## Create snake plot
sbn.lineplot(data=df_melt,x="RFM",y="Value",hue="Cluster",palette=palette,ax=ax)
ax.legend(title="Cluster",loc="upper right")
ax.set_title("Snake plot of clusters by RFM features")
add_border(ax)

## Heatmap of ratio RFM for each clusters
sbn.heatmap(data=relative_rfm,annot=True,linewidths=0.5,ax=ax2)
ax2.set_title("Heat map of clusters by RFM features")
add_border(ax2)

## Create pie chart
explode = (0.05, 0.05, 0.05, 0.05, 0.05)
ax3.pie(df_perc["Percent"], explode=explode,labels=df_perc['Cluster_Label'],autopct='%1.1f%%',
       shadow=True, startangle=90,colors=palette)
ax3.legend(title="Cluster",labels=df_perc["Cluster"],loc="upper right",bbox_to_anchor=(1.25,1.1))
ax3.set_title("Percentage of customers for each cluster")


plt.tight_layout()
save("AllCharts_RFM")
plt.show()

"""
## Export Dataset
"""

## Final dataset
df

## Table with percentage for each cluster
df_perc

## Create interactive table

### Dataset with clusters
render(df,to_file="df.html",buttons=['copy','csv'])


## Table with size and percentage of clusters
render(df_perc,to_file="df_perc.html",buttons=['copy','csv'])

!zip -r plots.zip plots

from google.colab import files

files.download("plots.zip")
