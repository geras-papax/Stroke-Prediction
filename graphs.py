import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import remove 
import mean
import linReg
import knear

sns.set()

df = pd.read_csv (r'C:\Users\makis\OneDrive\Υπολογιστής\Data Mining\healthcare-dataset-stroke-data\healthcare-dataset-stroke-data.csv')

data = df[(df['smoking_status']!="formerly smoked") | (df['smoking_status']!="smokes")]
data2 = df[(df['smoking_status']=="formerly smoked") | (df['smoking_status']=="smokes")]


# f, axes = plt.subplots(2, 3)
# sns.countplot(data=data, x='stroke', hue="gender", ax=axes[0][0])
# sns.countplot(data=data2, x="stroke", hue="gender", ax=axes[0][1])
# sns.lineplot(x="bmi", y ="stroke", hue="gender", data=df, ax=axes[0][2])
# sns.lineplot(x="age", y ="stroke", hue="gender", data=df, ax=axes[1][0])
# sns.countplot(x="hypertension", hue ="stroke", data=df, ax=axes[1][1])
# sns.countplot(x="heart_disease", hue ="stroke", data=df, ax=axes[1][2])

# f.suptitle('Searching for Relations', fontsize=16)

# axes[0][0].set_title("Non Smoking History - Stroke")
# axes[0][1].set_title("Smoking History - Stroke")
# axes[0][2].set_title("BMI - Stroke")
# axes[1][0].set_title("Age - Stroke")
# axes[1][1].set_title("Hypertension - Stroke")
# axes[1][2].set_title("Heart Disease - Stroke")
# plt.show()
stats = {'method':["precision score","f1 score","recall score"],
        'remove_stats':[remove.prec_score, remove.f1_score, remove.rec_score],
        'mean_stats':[mean.prec_score, mean.f1_score, mean.rec_score],
        'linear_stats':[linReg.prec_score, linReg.f1_score, linReg.rec_score],
        'KNN_stats':[knear.prec_score, knear.f1_score, knear.rec_score]
}
dstats= pd.DataFrame.from_dict(stats)
dstats= pd.melt(dstats, id_vars="method", var_name="stats_type", value_name="rate")
sns.lineplot(x=dstats['stats_type'], y=dstats["rate"],hue=dstats['method'],data=dstats)
plt.show()