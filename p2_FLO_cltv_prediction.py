########################################################################################################
# Dataset Info
########################################################################################################
# master_id: unique customer id
# order_channel: Which channel of the shopping platform is used (Android, iOS, Desktop, Mobile)
# last_order_channel: Channel of last purchase
# first_order_date: date of customers first purchase
# last_order_date: date of customers last purchase
# last_order_date_online: date of customers last online purchase
# last_order_date_offline: date of customers last offline purchase
# order_num_total_ever_online: total number of purchase in online
# order_num_total_ever_offline: total number of purchase in offline
# customer_value_total_ever_offline: total price of offline purchase
# customer_value_total_ever_online: total price of online purchase
# interested_in_categories_12: List of categories in which the customer shopped in the last 12 months
# 19.942 observation unit, 12 variable
########################################################################################################
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df_ = pd.read_csv("M3_crm_analytics/my_codes/datasets/flo_data_20k.csv")
df = df_.copy()
df.head()


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return round(low_limit), round(up_limit)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df.describe().T
replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

df["frequency"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["monetary"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

replace_with_thresholds(df, "frequency")
replace_with_thresholds(df, "monetary")

df.describe().T
df.info()
df.head()

date_cols = [col for col in df.columns if "date" in col]
df[date_cols] = df[date_cols].apply(pd.to_datetime)
df.info()
df.head()

# recency_weekly
# T_weekly
# frequency
# monetary_avg
today_date = df["last_order_date"].max() + dt.timedelta(2)

cltv = df[["master_id"]]
cltv["recency_weekly"].min()
cltv["recency_weekly"] = (df["last_order_date"] - df["first_order_date"]) / 7
cltv["recency_weekly"] = cltv["recency_weekly"].apply(lambda x: x.days)

cltv["T_weekly"] = (today_date - df["first_order_date"]) / 7
cltv["T_weekly"] = cltv["T_weekly"].apply(lambda x: x.days)

cltv["frequency"] = df[["frequency"]]

cltv["monetary"] = df["monetary"] / df["frequency"]

cltv.head()
cltv["frequency"] = cltv["frequency"].apply(lambda x: round(x))
cltv = cltv[cltv["recency_weekly"] > 0]
cltv.info()
cltv.describe().T
#####

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv["frequency"],
        cltv["recency_weekly"],
        cltv["T_weekly"])

cltv["exp_sales_3_month"] = bgf.predict(3 * 4,
                                        cltv["frequency"],
                                        cltv["recency_weekly"],
                                        cltv["T_weekly"])

cltv["exp_sales_6_month"] = bgf.predict(6 * 4,
                                        cltv["frequency"],
                                        cltv["recency_weekly"],
                                        cltv["T_weekly"])

cltv.head()

# gamma gamma


ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv["frequency"],
        cltv["monetary"])

cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv["frequency"],
                                                                    cltv["monetary"])

cltv["cltv"] = ggf.customer_lifetime_value(bgf,
                                           cltv["frequency"],
                                           cltv["recency_weekly"],
                                           cltv["T_weekly"],
                                           cltv["monetary"],
                                           time=6)


cltv.sort_values("cltv", ascending=False).head(20)


cltv["segment"] = pd.qcut(cltv["cltv"], 4, labels=["D", "C", "B", "A"])

cltv.groupby("segment").agg("mean")