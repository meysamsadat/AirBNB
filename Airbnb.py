import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(r'C:\Users\meysam-sadat\Desktop\airbnb_melbourn\cleansed_listings_dec18.csv',low_memory=False)
df.shape
# check how the data types are distributed.
df.dtypes.value_counts()
# Set the display properties so that we can inspect the data
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

#Drop columns that are not relevant to the problem. Example: URL, host picture etc.
#Find missing values for each column.
#Convert columns to their correct data type.
#Subset the dataset based on certain criterion. Ex: property_type = Apartment/House/Townhouse/Condo
#One-hot-encode the categorical variables
df_object_columns = df[df.select_dtypes(include=['object']).columns]
drop_object_cols = ['listing_url','last_scraped','picture_url','host_url',
             'host_name','host_since','host_location','host_about','host_thumbnail_url',
             'host_picture_url','street','city','state','zipcode','smart_location',
             'country_code','country','calendar_updated','calendar_last_scraped','first_review','last_review']
df_float_columns =df[df.select_dtypes(include=['float64']).columns]
df_int_columns = df[df.select_dtypes(include=['int64']).columns]
drop_int_float_cols = ['scrape_id','host_id']

drop_columns = drop_object_cols + drop_int_float_cols


print(f"Dropping {len(drop_columns)} columns")
df = df.drop(columns=drop_columns)
print('Shape of the dataset after dropping',df.shape)
df.shape

#Calculates missing value statistics for a given dataframe and
#returns a dataframe containing number of missing values per column
#and the percentage of values missing per column.
#arguments:
#df: the dataframe for which missing values need to be calculated.
def missing_statistics(df):

    missing_stats = df.isnull().sum().to_frame()
    missing_stats.columns = ['num_missing']
    missing_stats['pct_missing'] = np.round(100 * (missing_stats['num_missing'] / df.shape[0]))
    missing_stats.sort_values(by='num_missing', ascending=False, inplace=True)

    return missing_stats

num_missing = missing_statistics(df)
cols_to_drop = list(num_missing[num_missing.pct_missing > 20].index)
df_clean = df.drop(cols_to_drop, axis=1)
df_clean.shape
df_clean_object = df_clean.select_dtypes(['object']).head(10)
df_clean_object.columns
dummy_columns =['host_is_superhost','host_has_profile_pic', 'host_identity_verified',
       'suburb', 'is_location_exact', 'property_type', 'room_type', 'bed_type','has_availability',
       'requires_license', 'instant_bookable','cancellation_policy', 'require_guest_profile_picture',
       'require_guest_phone_verification']
df_clean[dummy_columns].dtypes
df_clean[dummy_columns] = df_clean[dummy_columns].apply(lambda x: x.astype('category'),axis=0)
#Let’s check how many unique values each of these columns have.
# This can be achieved using the pd.Series.nunique method
num_unique_values = df_clean[dummy_columns].apply(pd.Series.nunique, axis=0)

# Plot number of unique values for each label
num_unique_values.plot(kind='bar')
plt.xlabel('Labels')
plt.ylabel('Number of unique values')
suburb_numbers = df_clean.suburb.value_counts()
property_numbers = df_clean.property_type.value_counts()
print(suburb_numbers)
property_numbers.plot(kind='bar')
plt.xlabel('property_type')
plt.ylabel('numbers')
df_clean.to_csv(r'C:\Users\meysam-sadat\Desktop\airbnb_melbourn\cleaned_airbnb_meysam.csv',index=False)


df_airbnb = df_clean[df_clean.property_type.isin(['Apartment', 'House', 'Townhouse', 'Condominium'])]
df_airbnb = df_airbnb.copy()
df_airbnb.loc[:, 'property_type'] = df_airbnb.loc[:,'property_type'].cat.remove_unused_categories()

df_airbnb = df_airbnb[df_airbnb.suburb.isin(['Melbourne','Southbank','South Yarra','Saint Kilda'])]
df_airbnb = df_airbnb.copy()
df_airbnb.loc[:, 'suburb'] = df_airbnb.loc[:,'suburb'].cat.remove_unused_categories()
df_airbnb = df_airbnb.copy()

df_airbnb['suburb'].value_counts()



missing_df = missing_statistics(df_airbnb)
# collect all the columns which have missing values
cols_missing_values = list(missing_df[missing_df.num_missing > 0].index)

df_airbnb_missing_values = df_airbnb[cols_missing_values]

host_cols = list(df_airbnb_missing_values.columns[df_airbnb_missing_values.columns.str.contains('host')])
df_airbnb_missing_values[host_cols][df_airbnb_missing_values.host_identity_verified.isnull()]
#Handling Missing Values
#For columns containing text, we will be replacing them with an empty string.
#For categorical columns, we will be replacing missing values with the mode.
#For continuous columns, we will be replacing the missing values with the median
df_airbnb_clean = df_airbnb.copy(deep=True)
#df.loc[df.column.isna().copy(),'column'] = '' good method for cleaning
df_airbnb_clean.loc[df_airbnb_clean.summary.isna().copy(), 'summary'] = ''
df_airbnb_clean.loc[df_airbnb_clean.description.isna().copy(), 'description'] = ''
df_airbnb_clean.loc[df_airbnb_clean.name.isna().copy(), 'name'] = ''

from sklearn.impute import SimpleImputer
df_airbnb_clean.isnull().sum()
category_missing_cols = ['host_has_profile_pic','host_identity_verified']

float_missing_cols = ['bathrooms', 'beds', 'bedrooms']


def replace_missing_values(cols, df):
    '''
        Takes a list of columns and a dataframe and imputes based on
        the column type. If it is object type, then most_frequent value
        is used for imputation. If it is a float/int type, then the median
        value is used for imputation.
        arguments:
            cols: list of columns
            df : dataframe containing these columns.
        returns:
            df: the imputed dataframe
    '''
    for col in cols:
        if type(df[col].dtype) is pd.core.dtypes.dtypes.CategoricalDtype:
            print("Imputing {} column with most frequent value".format(col))
            mode_imputer = SimpleImputer(strategy='most_frequent')
            df.loc[:, col] = mode_imputer.fit_transform(df[[col]])
        elif df[col].dtype == 'float64' or df[col].dtype == 'int64':
            print("Imputing {} column with median value".format(col))
            median_imputer = SimpleImputer(strategy='median')
            df.loc[:, col] = median_imputer.fit_transform(df[[col]])
        else:
            raise ValueError("Invalid column type")

    return df

df_airbnb_clean = replace_missing_values(float_missing_cols, df_airbnb_clean)
df_airbnb_clean = replace_missing_values(category_missing_cols, df_airbnb_clean)

df_airbnb_clean = df_airbnb_clean.fillna(method='ffill')
df_airbnb_clean.isnull().sum()
df_airbnb_clean.shape
df_airbnb_clean.columns

categorical_types = ['host_is_superhost','host_has_profile_pic', 'host_identity_verified',
       'suburb','is_location_exact', 'property_type','has_availability',
       'room_type','bed_type','guests_included', 'extra_people','has_availability','requires_license',
       'instant_bookable', 'cancellation_policy','require_guest_profile_picture', 'require_guest_phone_verification']

category_one_hot_encoding = pd.get_dummies(df_airbnb_clean[categorical_types])
category_one_hot_encoding.shape
df_airbnb_clean = pd.concat([df_airbnb_clean,category_one_hot_encoding],axis=1)
df_airbnb_clean = df_airbnb_clean.drop(categorical_types,axis=1)

df_airbnb_clean.shape
# End of cleaning
df_airbnb.to_csv(r'C:\Users\meysam-sadat\Desktop\airbnb_melbourn\Final_cleaned_airbnb_by_meysam.csv')
df_airbnb.isnull().sum()
df_airbnb.dropna(axis=0,inplace=True)
df_airbnb.isnull().sum()
df_airbnb.shape
df_airbnb_clean.columns
df_airbnb_clean = df_airbnb_clean[['price','latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms',
       'beds','minimum_nights', 'maximum_nights',
       'availability_30', 'availability_60', 'availability_90',
       'availability_365', 'number_of_reviews',
       'calculated_host_listings_count', 'host_is_superhost_f',
       'host_is_superhost_t', 'host_has_profile_pic_f',
       'host_has_profile_pic_t', 'host_identity_verified_f',
       'host_identity_verified_t', 'suburb_Melbourne', 'suburb_Saint Kilda',
       'suburb_South Yarra', 'suburb_Southbank', 'is_location_exact_f',
       'is_location_exact_t', 'property_type_Apartment',
       'property_type_Condominium', 'property_type_House',
       'property_type_Townhouse', 'has_availability_t',
       'room_type_Entire home/apt', 'room_type_Private room',
       'room_type_Shared room', 'bed_type_Airbed', 'bed_type_Couch',
       'bed_type_Futon', 'bed_type_Pull-out Sofa', 'bed_type_Real Bed',
       'has_availability_t', 'requires_license_f', 'instant_bookable_f',
       'instant_bookable_t', 'cancellation_policy_flexible',
       'cancellation_policy_moderate', 'cancellation_policy_strict',
       'cancellation_policy_strict_14_with_grace_period',
       'cancellation_policy_super_strict_30',
       'cancellation_policy_super_strict_60',
       'require_guest_profile_picture_f', 'require_guest_profile_picture_t',
       'require_guest_phone_verification_f',
       'require_guest_phone_verification_t']]

x = df_airbnb_clean.iloc[:,1:]
y = df_airbnb_clean['price']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LassoCV,ElasticNetCV,RidgeCV
from sklearn.preprocessing import StandardScaler
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.metrics import mean_squared_error
scale = StandardScaler()
scale.fit(x_train)
x_train_scaled = scale.transform(x_train)
x_test_scaled = scale.transform(x_test)
baseline = y_train.median()
baseline_error = np.sqrt(mean_squared_error(y_pred=np.ones_like(y_test) * baseline,y_true=y_test))

lr = LinearRegression()
alphas = [1000,100,50,20,10,1,0.1,0.01]
l1_rtios = [0.001,0.01,0.05,0.1,0.3,0.5,0.7,0.9]
ridge = RidgeCV(alphas=alphas)
lasso = LassoCV(alphas=alphas,max_iter=10000)
elastic = ElasticNetCV(alphas=alphas,l1_ratio=l1_rtios)

for model,name in zip([lr,ridge,lasso,elastic],['LinarRegression','Ridge','Lasso','ElasticNet']):
    model.fit(x_train_scaled,y_train)
    y_pred_train = model.predict(x_train_scaled)
    mrse_train = np.sqrt(mean_squared_error(y_pred=y_pred_train,y_true=y_train))
    y_pred = model.predict(x_test_scaled)
    mrse_test = np.sqrt(mean_squared_error(y_pred=y_pred,y_true=y_test))
    best_alpha = ''
    if name != 'LinarRegression':
        best_alpha = 'best alpha:'+ str(model.alpha_)
    best_l1 = ''
    if name == 'ElasticNet':
        best_l1 = 'best l1'+ str(model.l1_ratio_)
    print(name + 'mrse_train:'+str(mrse_train)+ ',mrse_test:'+ str(mrse_test) + best_alpha + best_l1)

print(lasso.coef_)

order = np.argsort(np.abs(lasso.coef_))[::-1]
for i in order:
    coef_ = lasso.coef_[i]
    if coef_ > 0:
        print(x.columns[i]+ ',' +str(lasso.coef_[i]))

y_pred_train = lasso.predict(x_train_scaled)
diff = y_train - y_pred_train
plt.figure(figsize=(15,8))
plt.scatter(np.arange(len(diff)),diff)

x_train_diff = x_train[np.abs(diff) > 100].describe()

high_error = x_train[np.abs(diff) > 80]
print('Size high Error'+ str(len(high_error)))

low_error = x_train[np.abs(diff) < 10]
print('Size Low Error' + str(len(low_error)))

for c in high_error.columns:
    plt.figure(figsize=(7,3))
    plt.subplot(121)
    plt.hist(low_error[c],color='b')
    plt.title(c +'low_error')
    plt.subplot(122)
    plt.hist(high_error[c],color='r')
    plt.title(c + 'high_error')