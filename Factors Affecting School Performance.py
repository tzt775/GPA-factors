#!/usr/bin/env python
# coding: utf-8

# # U.S. High School GPA Analysis

# In recent years especially, the U.S. education system has come under 
# extreme scrutiny for low performance in traditional subjects. <br> While many
# studies focus on the schools themselves, it is also important to look 
# at factors affecting a students outside of the classroom.
# 
# "Education and Poverty: Confronting the Evidence\" by Helen F.
# Ladd is a 2012 article that examines many of the reforms to the school <br>
# system of the Obama adminstration. Her claim was the creation of
# policies to evaluate and alter education are pointless because
# many <br> of them did not take into account how significant
# income disparities are to individual performance.

# Using household condition and state spending on kids data from the 
# Urban Institute, we will attempt to find which non-school related <br> factors 
# impact how students do in school.

# In[1]:


import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.othermod.betareg import BetaModel
from statsmodels.tools.eval_measures import rmse
from math import sqrt
import scipy.stats as ss
import plotly.graph_objects as go

STATE = 219


# ## Functions

# In[2]:


def mape(actual, predicted):
    n = len(actual)
    mape = np.sum(abs((actual - predicted)/actual))/n * 100
    return round(mape, 2)


def gpa_calc(state):
    """
    calculates average state GPA
    Args:
        state (string) = name of state in state dictionary
    Returns:
        (float) = estimated state GPA
    """
    apl, a, amin, b, c, less = state_gpa[state]
    return round(((apl*4.0) + (a*3.7) + (amin*3.3) + (b*3.0) + (c*2.0) + (less*1.0)), 2)


# In[3]:


# Spending (public spending in $1,000's):
# HCD = housing and community development
# TANFbasic = temporary assistance for needy families
# othercashserv = other cash assit. services/social services
# unemp = uneployment compensation benefits
# lib = libraries
# parkrec = parks and recreation
# SNAP = supplemental nutrition assistance program

def filter_excel(sheet):
    """
    read in, filter and calculate average 2014-2016 spending
    Args:
        sheet = sheet of interest from State-by-State Spending on kids
    Returns:
        df (dataframe) = spending by state
    """
    df = pd.read_excel('State-by-State Spending on Kids.xlsx', sheet_name = sheet)
    df = df[['state', '2014', '2015', '2016']]
    df['2014-2016 Avg {}'.format(sheet)] = round(df.mean(axis = 1, numeric_only = True), 2)
    df = df.drop(['2014', '2015', '2016'], axis = 1)
    df.rename(columns = {'state': 'State'}, inplace = True)
    return df

types_of_spending = ["HCD", "PK12ed", "highered", "TANFbasic", "othercashserv", 
                     "SNAP", "unemp", "lib", "parkrec"]

types_of_spending = list(map(filter_excel, types_of_spending))
spending = reduce(lambda  a, b: pd.merge(a, b, on = ['State'], how = 'outer'), 
                  types_of_spending)

spending = spending.loc[spending["State"] != "District of Columbia"]
#spending.head()


# ## Importing and Cleaning Data

# In[4]:


NHGIS = pd.read_excel('NHGIS_District_data.xlsx')

child_count = NHGIS[['State', 'Children 5-17 (SAIPE Estimate)']]
NHGIS = NHGIS.drop(['Geographic School District', 'School ID', 
                    'Children 5-17 (SAIPE Estimate)'], axis = 1)

# proportion children = proportion of all US children in state
# columns = avg. percent of households affected in each state
NHGIS_avg = NHGIS.groupby(by = 'State').mean()
NHGIS_avg['Sum State Children'] = child_count.groupby(by = 'State').sum()
total_kids = NHGIS_avg['Sum State Children'].sum()
NHGIS_avg['Proportion Children'] = [int(state_total)/total_kids for state_total 
                                    in NHGIS_avg['Sum State Children']]
factors = pd.merge(NHGIS_avg, spending, on = 'State')
#factors.head()


# In[5]:


# spending per child to account for population differences

per_child = ["{} per child".format(col) for col in factors.iloc[:, 10:19]]
for col in per_child:
    factors[col] = factors[" ".join(col.split(" ")[:-2])]/factors["Sum State Children"]
#factors.head()


# ### Calculate GPA Estimates

# In[6]:


# add GPAs (2017)
# A+ (97–100) = 4.0
# A (93–96) = 3.7
# A- (90–92) = 3.3
# B (80–89) = 3.0
# C (70–79) = 2.0
# D, E, or F (below70) = 1.0



state_gpa = {"Alabama":[.2, .32, .18, .25, .03, 0], "Alaska":[.09, .19, .18, .36, .10, .01], 
             "Arizona":[.08, .23, .19, .37, .08, 0], "Arkansas":[.20, .36, .17, .22, .03, 0], 
             "California":[.05, .16, .18, .46, .10, 0], "Colorado":[.14, .28, .20, .27, .04, 0],
             "Connecticut":[.03, .12, .17, .46, .12, .01], "Delaware":[.05, .16, .16, .44, .13, .01], 
             "Florida":[.05, .15, .15, .45, .13, .01], "Georgia":[.06, .18, .19, .46, .08, 0], 
             "Hawaii":[.06, .20, .21, .41, .08, 0], "Idaho":[.04, .17, .15, .36, .14, .01], 
             "Illinois":[.09, .21, .15, .32, .13, .01], "Indiana":[.06, .18, .17, .44, .12, .01], 
             "Iowa":[.22, .38, .15, .18, .03, 0], "Kansas":[.20, .36, .18, .22, .02, 0], 
             "Kentucky":[.23, .37, .17, .18, .02, 0], "Louisiana":[.22, .31, .18, .24, .03, 0], 
             "Maine":[.05, .16, .16, .40, .09, .01], "Maryland":[.04, .15, .17, .46, .13, .01], 
             "Massachusetts":[.03, .14, .19, .48, .10, 0], "Michigan":[.05, .14, .14, .35, .19, .03], 
             "Minnesota":[.20, .36, .18, .21, .01, 0], "Mississippi":[.34, .35, .14, .14, .01, 0], 
             "Missouri":[.21, .33, .19, .21, .03, 0], "Montana":[.12, .32, .23, .27, .02, 0], 
             "Nebraska":[.25, .35, .18, .16, .02, 0], "Nevada":[.07, .21, .22, .41, .06, 0], 
             "New Hampshire":[.03, .12, .15, .43, .11, .01], "New Jersey":[.05, .17, .19, .46, .09, 0], 
             "New Mexico":[.11, .25, .22, .32, .06, 0], "New York":[.06, .17, .16, .43, .12, .01], 
             "North Carolina":[.09, .27, .20, .35, .06, 0], "North Dakota":[.28, .40, .14, .13, .03, 0], 
             "Ohio":[.12, .27, .19, .30, .07, 0], "Oklahoma":[.14, .24, .13, .29, .11, .01], 
             "Oregon":[.09, .23, .21, .37, .06, 0], "Pennsylvania":[.08, .22, .20, .39, .06, 0], 
             "Rhode Island":[.03, .15, .19, .48, .09, 0], "South Carolina":[.08, .27, .22, .37, .04, 0], 
             "South Dakota":[.23, .32, .14, .22, .03, 0], "Tennessee":[.21, .33, .18, .24, .02, 0], 
             "Texas":[.06, .16, .19, .47, .07, 0], "Utah":[.24, .33, .16, .20, .02, 0], 
             "Vermont":[.05, .20, .21, .43, .06, 0], "Virginia":[.07, .20, .18, .41, .09, 0], 
             "Washington":[.06, .18, .18, .40, .10, .01], "West Virginia":[.22, .31, .17, .23, .03, 0], 
             "Wisconsin":[.23, .38, .16, .15, .03, 0], "Wyoming":[.21, .40, .17, .19, .01, 0]}



factors["Est GPA"] = pd.DataFrame(map(gpa_calc, state_gpa))


# ### Drop Original Average Spending Columns and Sum State Childen

# In[7]:


factors = factors.drop(["2014-2016 Avg HCD", "2014-2016 Avg PK12ed", "2014-2016 Avg highered", 
                        "2014-2016 Avg TANFbasic", "2014-2016 Avg othercashserv", 
                        "2014-2016 Avg SNAP", "2014-2016 Avg unemp", "2014-2016 Avg lib", 
                        "2014-2016 Avg parkrec", "Sum State Children"], axis = 1)
#factors.info()


# ## Visualize Data

# In[8]:


est_gpa = px.histogram(factors, x = "Est GPA", title = "US 2017 GPA Estimates", 
                       labels = {"Est GPA":"Estimated GPA"})
gpa_median = round(factors["Est GPA"].median(), 2)
#est_gpa.add_vline(x = gpa_median, annotation_text = "Median {}".format(gpa_median), 
                  #annotation_position = "top left")
gpa_mean = round(factors["Est GPA"].mean(), 2)
#est_gpa.add_vline(x = gpa_mean, line_color = "white", annotation_text = "Average {}".format(gpa_mean))
#est_gpa.show()

#est_gpa.write_image("Results/2017 est GPA.png")


# There are 16 states below the bar containing the median, and 19 above. Bars below median have slightly larger range, 0.59 vs 0.39, <br> but only 1 state in 2.4-2.59 range, which may be an outlier. Since the mean of 3.12 is very close to the median of 3.105, we can <br> assume GPA is normally distributed.

# In[9]:


pie_factors = factors.copy()
pie_factors.loc[pie_factors['Proportion Children'] < .02, 'State'] = 'Other'
# other = states with less than 2% of total US children

child_distribution = px.pie(pie_factors, values = 'Proportion Children', 
                            names = 'State', title = 'U.S. School Aged Children by State')
#child_distribution.show()

#child_distribution.write_image("Results/children by state.png")


# 14 states are estimated to have 2/3 of the school aged children in the U.S, with almost 15% of children located in California alone. <br> Since different states have different amounts of children

# In[10]:


avg_pk12 = px.bar(factors, 
                  x = ["2014-2016 Avg PK12ed per child", "2014-2016 Avg highered per child"], 
                  y = "State", barmode = 'group', 
                  title = "Average PK-12 vs Post Secondary Spending per Child", 
                  labels = {"variable" : "Spending Type", "value" : "Spending Per Child (in $1,000s)"})
#avg_pk12.update_layout(yaxis = {'categoryorder':'total descending'})
#avg_pk12.show()

#avg_pk12.write_image("Results/per child education spending.png")


# There does not appear to be any noticible relationship between average per child spending on K-12 and higher education. There are <br> likely unknown factors influencing state education spending, such as number of public schools and proportion of students attending <br> college from in vs out of state.

# In[11]:


ne = ["Connecticut", "New Hampshire", "Vermont", "Rhode Island", "New York", "New Jersey", 
      "Maine", "Massachusetts", "Pennsylvania"]
south = ["Texas", "Louisiana", "Oklahoma", "Tennesee", "Kentucky", "Georgia", "Alabama", 
         "Florida", "West Virginia", "Virginia", "Maryland", "Delaware", "South Carolina", 
         "Mississippi", "Arkansas", "North Carolina"]
mw = ["Missouri", "Kansas", "Nebraska", "North Dakota", "South Dakota", "Wisconsin", 
      "Minnisota", "Ohio", "Iowa", "Michigan", "Indiana", "Illinois"]

factors["Region"] = factors["State"].apply(lambda x: "Northeast" if x in ne else 
                                           ("Midwest" if x in mw else 
                                            ("South" if x in south else
                                             ("West"))))

region_plot = px.scatter(factors, x = "State", y = "Est GPA", size = "Proportion Children", 
                         color = "Region", hover_name = "State", size_max = 35, 
                         title = "State GPA by Region")
#region_plot.show()

#region_plot.write_image("Results/region GPA.png")


# Northeastern states tend to have the lowest estimated GPAs, with the highest being 3.08. 
# This may be because of the 10 states that <br> required the SAT for high school graduation in 2017, the Northeast has 3 of them (Connecticut, New Hampshire, and Maine) (Peck). 
# - States that require the SAT also tend to be among the lowest in their region overall
#  - Delaware second lowest GPA in South
#  - Michigan lowest GPA in Midwest
#  - Connecticut, New Hampshire and Maine lowest GPAs in Northeast

# In[12]:


#factors.Region.value_counts()


# ### Rename Columns

# In[13]:


factors.columns = ['State', 'Poverty', 'Single_Parent', 'Vulnerable_Job', 'Crowded_Conditions', 
                   'No_Internet', 'Disabled_Children', 'Linguistically_Isolated_Children', 
                   'Proportion_Children', 'Avg_HCD', 'Avg_PK12ed', 'Avg_highered','Avg_TANFbasic', 
                   'Avg_othercashserv', 'Avg_SNAP', 'Avg_unemp', 'Avg_lib', 'Avg_parkrec', 
                   'Est GPA 2017', 'Region']
factors = pd.get_dummies(factors, columns = ["Region"])
factors.head()


# In[14]:


# Assuming normality - outliers
m = factors["Est GPA 2017"].mean()
st = factors["Est GPA 2017"].std()
print("Lower Bound: {}\nUpper Bound: {}".format(m - 2*st, m + 2*st))

#only 1 state with GPA est below 2.6
outlier = np.where(factors["Est GPA 2017"] < m - 2*st)
print(factors.iloc[outlier[0], 0])


# In[15]:


col_sub = factors.columns.drop(["State", "Est GPA 2017"])

#for col in col_sub[0:17]:
    #plot = px.scatter(x = factors[col], y = factors["Est GPA 2017"], labels = {"x":col, "y":"GPA Est"})
    #plot.update_layout(title = col)
    #plot.show()


# Several variables have some kind of correlation with Est GPA 2017, but it is difficult to determine if those relationships are linear.

# ### Split Data

# In[16]:


X = factors.drop(["State", "Est GPA 2017"], axis = 1)
y = factors[["Est GPA 2017"]]


# In[17]:


#abs(X.corr()) > 0.90
# Poverty and No_internet highly correlated - cannot include in same model


# In[18]:


X = X.drop(["Poverty"], axis = 1)
#X.info()


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = STATE)


# ## Models

# ### OLS

# In[20]:


X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train)
ols_1 = model.fit()
ols_1.summary()


# In[21]:


X_test = sm.add_constant(X_test)
ols1_pred = ols_1.predict(X_test)

ols_rsq = 0.74
ols_aic = ols_1.aic
ols_bic = ols_1.bic

ols_rmse = rmse(y_test["Est GPA 2017"], ols1_pred)
print(ols_rmse)

ols_mape = mape(y_test["Est GPA 2017"], ols1_pred)
ols_mape


# In[22]:


vs_plot = go.Figure()
vs_plot.add_trace(go.Scatter(x = y_test["Est GPA 2017"], y = ols1_pred, 
                             name = "GPA", mode = "markers"))

vs_plot.add_trace(go.Scatter(x = [2.6, 3.8], y = [2.6, 3.8],
                             line = dict(color = 'black', width = 2), name = "Middle Line"))

vs_plot.update_layout(title = "OLS: Actual vs Predicted GPA", xaxis_title = "Actual", 
                      yaxis_title = "Predicted")
vs_plot.show()

#vs_plot.write_image("Results/ols actual vs predicted.png")


# Most points on the Actual vs Predicted plot are above the diagonal line, meaning the OLS model is regularly overpredicting the GPA.

# ### Check Residuals for Patterns

# In[23]:


influence = ols_1.get_influence()

leverage = influence.hat_matrix_diag

#cooks_d = influence.cooks_distance

standardized_residuals = influence.resid_studentized_internal

fig, axes = plt.subplots(nrows = 2, ncols = 2)
fig.tight_layout(pad = 3.5)
plt.style.use('seaborn')

axes[0, 0].scatter(x = leverage, y = standardized_residuals)
axes[0, 0].set_title("Leverage vs Standardized Residuals")

sm.qqplot(standardized_residuals, dist = ss.t, fit = True, line = '45', ax = axes[0, 1])
axes[0, 1].set_title("Normal Q-Q Plot")

axes[1, 0].scatter(x = y_train["Est GPA 2017"], y = standardized_residuals)
axes[1, 0].set_title("Actual vs Standardized Residuals")

axes[1, 1].hist(standardized_residuals, bins = 6)
axes[1, 1].set_title("Histogram of Standardized Residuals")

plt.show()

#fig.savefig("Results/ols residuals.png")


# From the above plots, there are no points with extremely high leverages, the residuals are approximately normal (with a slight left skew <br> which could be the result of having a small sample size), and there is a strange curvature in the actual vs standardized residual plot, <br> so OLS may not be capturing some underlying pattern.

# ### Transform GPA

# In[24]:


log_gpa = np.log(y_train["Est GPA 2017"])
# still left skewed
sqrt_gpa = np.sqrt(y_train["Est GPA 2017"])
# flat or bimodal (depending on # bins)
cubrt_gpa = np.cbrt(y_train["Est GPA 2017"])
# bimodal
recip_gpa = np.reciprocal(y_train["Est GPA 2017"])
# heavily right skewed

reflect_gpa = (y_train["Est GPA 2017"]*-1+3.6)
# slightly right skewed

y_transform = (y_train["Est GPA 2017"]*0.25)
# scaled slightly left skewed

#px.histogram(y_transform)


# ### Beta Model with Data Transformed
# 
# Justification: Beta distribution can be used to model left skewed data
# 
# Transformation: y_transform = y*0.25
# - 0.25 scales GPA to fit within 0-1 range (bounds of Beta)

# In[25]:


beta = BetaModel(y_transform, X_train)
beta_1 = beta.fit()
beta_1.summary()


# In[26]:


y_test_transform = (y_test["Est GPA 2017"]*.25)

beta_pred = beta_1.predict(X_test)

beta_aic = beta_1.aic
beta_bic = beta_1.bic

beta_rmse = rmse(y_test_transform, beta_pred)
print(beta_rmse)

beta_mape = mape(y_test_transform, beta_pred)
beta_mape


# In[27]:


vs_plot_beta = go.Figure()
vs_plot_beta.add_trace(go.Scatter(x = y_test_transform, y = beta_pred, 
                                  name = "GPA", mode = "markers"))

vs_plot_beta.add_trace(go.Scatter(x = [0.6, 0.9], y = [0.6, 0.9],
                                  line = dict(color = 'black', width = 2), name = "Middle Line"))

vs_plot_beta.update_layout(title = "Beta: Actual vs Predicted GPA", xaxis_title = "Actual", 
                           yaxis_title = "Predicted")
vs_plot_beta.show()

#vs_plot_beta.write_image("Results/beta actual vs predicted.png")


# ### GLM with Transformed Predictor
# 
# Justification:
# Multiple predictor variables not linearly correlated with GPA, GPA may not be normally 
# distributed, and OLS residuals still <br> show potential left skew.
# 
# Transformation: reflect_gpa = y*-1 + 3.6
# - Inverse of GPA to make data slightly right skewed to try other distributions, ie. Gamma
# - Add 3.6 because Gamma requires all values greater than 0

# In[28]:


gamma = sm.GLM(reflect_gpa, X_train, family = sm.families.Gamma(sm.families.links.Log()))
glmg = gamma.fit()
glmg.summary()


# In[29]:


reflect_test = (y_test["Est GPA 2017"]*-1+3.6)

glmg_pred = glmg.predict(X_test)

glmg_rsq = 0.903

glmg_rmse = rmse(reflect_test, glmg_pred)
print(glmg_rmse)

glmg_mape = mape(reflect_test, glmg_pred)
glmg_mape


# In[30]:


vs_plot_glmg = go.Figure()
vs_plot_glmg.add_trace(go.Scatter(x = reflect_test, y = glmg_pred, 
                                  name = "GPA", mode = "markers"))

vs_plot_glmg.add_trace(go.Scatter(x = [0., 1.5], y = [0., 1.5],
                                  line = dict(color = 'black', width = 2), name = "Middle Line"))

vs_plot_glmg.update_layout(title = "GLM Gamma: Actual vs Predicted GPA", xaxis_title = "Actual", 
                           yaxis_title = "Predicted")
vs_plot_glmg.show()

#vs_plot_glmg.write_image("Results/glmg actual vs predicted.png")


# So far, the beta model using the scaled GPA is the best performing with the smallest MAPE 8.52% compared to  9.29% and 56.98%, and <br> smaller AIC and BIC than OLS model.
# 
# Gamma GLM performs very well on the training data, a Pseudo-Rsq of 0.903, but very poorly on the test data where the RMSE was <br> slightly larger than the OLS, but the MAPE was close to 60%.
# 
# Since the MAPE of the OLS and beta models were very similar, we will refit both the models using a subset of variables.

# ## Model Tuning

# ### Subset for Statistically Significant Variables 
# 
# (p-value less than 0.10 from previous models - except Avg_unemp: inclusion breaks beta model)

# In[31]:


X_train = X_train[["const", "Crowded_Conditions", "Linguistically_Isolated_Children", 
                   "Avg_highered", "Avg_othercashserv", "Avg_lib", "Region_Midwest", 
                   "Region_Northeast", "Region_South", "Region_West"]]

X_test = X_test[["const", "Crowded_Conditions", "Linguistically_Isolated_Children", 
                 "Avg_highered", "Avg_othercashserv", "Avg_lib", "Region_Midwest", 
                 "Region_Northeast", "Region_South", "Region_West"]]


# ### OLS with Subset

# In[32]:


model2 = sm.OLS(y_train, X_train)
ols_sub = model2.fit()
ols_sub.summary()


# In[33]:


ols_sub_pred = ols_sub.predict(X_test)

ols_sub_rsq = 0.627

ols_sub_aic = ols_sub.aic
ols_sub_bic = ols_sub.bic

ols_sub_rmse = rmse(y_test["Est GPA 2017"], ols_sub_pred)
print(ols_sub_rmse)

ols_sub_mape = mape(y_test["Est GPA 2017"], ols_sub_pred)
ols_sub_mape


# In[34]:


vs_plot_ols_sub = go.Figure()
vs_plot_ols_sub.add_trace(go.Scatter(x = y_test["Est GPA 2017"], y = ols_sub_pred, 
                                     name = "GPA", mode = "markers"))

vs_plot_ols_sub.add_trace(go.Scatter(x = [2.6, 3.8], y = [2.6, 3.8],
                                     line = dict(color = 'black', width = 2), 
                                     name = "Middle Line"))

vs_plot_ols_sub.update_layout(title = "OLS Sub: Actual vs Predicted GPA", xaxis_title = "Actual", 
                              yaxis_title = "Predicted")
vs_plot_ols_sub.show()

#vs_plot_ols_sub.write_image("Results/ols sub actual vs predicted.png")


# The R-sq value dropped from 0.74 to 0.63, but AIC and BIC decreased, as well as MAPE and MSE, 
# so this reduced OLS model fits the <br> training data more than the full OLS model, and 
# performs better on unseen data.

# ### Beta on Subset of Variables

# In[35]:


beta2 = BetaModel(y_transform, X_train)
beta_sub = beta2.fit()
beta_sub.summary()


# In[36]:


beta_sub_pred = beta_sub.predict(X_test)

beta_sub_aic = beta_sub.aic
beta_sub_bic = beta_sub.bic

beta_sub_rmse = rmse(y_test_transform, beta_sub_pred)
print(beta_sub_rmse)

beta_sub_mape = mape(y_test_transform, beta_sub_pred)
beta_sub_mape


# In[37]:


vs_plot_beta_sub = go.Figure()
vs_plot_beta_sub.add_trace(go.Scatter(x = y_test_transform, y = beta_sub_pred, 
                                      name = "GPA", mode = "markers"))

vs_plot_beta_sub.add_trace(go.Scatter(x = [0.6, 0.9], y = [0.6, 0.9],
                                      line = dict(color = 'black', width = 2), name = "Middle Line"))

vs_plot_beta_sub.update_layout(title = "Beta Sub: Actual vs Predicted GPA", xaxis_title = "Actual", 
                               yaxis_title = "Predicted")
vs_plot_beta_sub.show()

#vs_plot_beta_sub.write_image("Results/beta sub actual vs predicted.png")


# ### Check for Influential Points

# In[38]:


l = ols_sub.get_influence().summary_frame().sort_values("cooks_d", ascending = False)
#print(l)

n = len(l)
print(np.where(l.cooks_d > 0.5))


# In[39]:


#influence_plot = ols_sub.get_influence().plot_influence(external = True, criterion = "Cooks")
#print(influence_plot)

#influence_plot.savefig("Results/ols sub influence plot.png")

influential = [1, 4, 11, 18]
factors.iloc[influential]


# 1 = Alaska, 4 = California, 11 = Idaho, 18 = Maine
# 
# From the OLS on the statistically significant variables, Alaska and California have a high 
# leverage, and Idaho and Maine have large <br> residuals, so these 4 states could be influencing 
# the fit of the model.

# ## Conclusion

# ### Results

# In[40]:


models = ["OLS", "Beta", "OLS Subset", "Beta Subset"]
rsqs = [ols_rsq, 0, ols_sub_rsq, 0]
aics = [ols_aic, beta_aic, ols_sub_aic, beta_sub_aic]
bics = [ols_bic, beta_bic, ols_sub_bic, beta_sub_bic]
rmses = [ols_rmse, beta_rmse, ols_sub_rmse, beta_sub_rmse]
mapes = [ols_mape, beta_mape, ols_sub_mape, beta_sub_mape]


# In[41]:


outcome = pd.DataFrame({"Model":models, "R-Sq":rsqs, "AIC":aics, "BIC":bics, 
                        "RMSE":rmses, "MAPE":mapes})
outcome


# In[42]:


fig, axes = plt.subplots(nrows = 1, ncols = 4)
fig.tight_layout(pad = 5.5)
plt.style.use('seaborn')

space = np.arange(4)
width = 0.5

plt.setp(axes, xticks = space, xticklabels = models, xlabel = "Models")

axes[0].bar(models, aics, width, color = "blue")
axes[0].set_xticklabels(models, rotation = 45, ha = "right")
axes[0].set_title("AIC")

axes[1].bar(models, bics, width, color = "black")
axes[1].set_xticklabels(models, rotation = 45, ha = "right")
axes[1].set_title("BIC")

axes[2].bar(models, rmses, width, color = 'navy')
axes[2].set_xticklabels(models, rotation = 45, ha = "right")
axes[2].set_title("RMSE")

axes[3].bar(models, mapes, width, color = 'green')
axes[3].set_xticklabels(models, rotation = 45, ha = "right")
axes[3].set_title("MAPE")

#plt.show()

#fig.savefig("Results/model eval.png")


# The Beta model using a subset of significant variables is the best performing model on both the training and testing data. Both of the <br> Beta models have much larger AIC and BIC values than the OLS models, and the Beta subset model has the smallest MAPE of 7.40%, <br> so it can predict GPA the best.
# 
# There are likely other related school district or state level variables that can explain school performance as well or better than these variables. 

# ### End Notes

# 1. Spending data is a record of public spending between 1997 - 2016 
# (only using 2014 - 2016 data), and household condition <br> 
# data is % estimate of affected households of 13,000 U.S. school districts from 2014 - 2018.
# 
# 
# 2. Spending categories are averages of 2014-2016 spending because many students take 
# the SAT junior year of highschool, <br> 
# which would correspond to 2017 for those starting 
# highschool in 2014.  
# 
# 
# 3. Being able to weigh the spending by district instead of state may have produced more 
# accurate results. The individual affects <br> of the districts were muddled when taking the 
# averages, unless rates of quality of life factors were consistent throughout the <br> entire 
# state, which is very unlikely.
# 
# 
# 4. State GPAs were estimated due to limited information available. I used the GPA 
# information listed under the College Board 2017 <br> SAT test taker information by state.
# 
# 
# 5. This method introduces some level of bias, as state laws surrounding taking the SAT 
# vary and the GPA calculations were based <br> solely off SAT test takers, there may be a 
# significant difference in the GPAs of those who took the SAT from those who did not, <br> 
# and may affect the accuracy of the model applied to the actual U.S. population of
# school aged children.
# 
# 
# 6. The "Poverty" variable was chosen to be dropped before modeling because it had a high chance of multicollinearity with "No_Internet", <br> but of the two, likely includes other unseen effects that would make it more difficult to interpret as a single factor. 
#     - And while the poverty line is the same across all states, the cost of living is not, so living below poverty may look very <br> different in each state, depending on members of household and their individual needs.
# 
# 
# 7. RMSE for the Beta models can only be compared with eachother, since the Beta models use a scaled GPA.
# 
# 
# 8. Due to the nature of the data, we assume that the household and spending information of each state is unique and based on many <br> factors affecting the state individually, so outliers and high leverage points can be left in the data.

# ### Sources

# Ladd, H. F. "Education and Poverty: Confronting the Evidence". Wiley Online Library, 28 Mar. 2012,<br>
#     https://onlinelibrary.wiley.com/doi/abs/10.1002/pam.21615
# 
# NHGIS. "Household Conditions by Geographic School District". Urban Institute.<br>
#     https://datacatalog.urban.org/dataset/household-conditions-geographic-school-district
#   
# Peck, Timothy. “List of States That Require the SAT.” CollegeVine Blog, 19 Jan. 2021, 
#     https://blog.collegevine.com/states-that-require-sat
# 
# "State-by-State Spending on Kids Dataset." Urban Institute. 
#     https://datacatalog.urban.org/dataset/state-state-spending-kids-dataset

# In[ ]:




