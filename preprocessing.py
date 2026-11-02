"""The exact feature engineering and preprocessing used by UNETI18 training.
Derived from cell 04 of UNETI18_Vong2_ChotNop_DA_SUA_LOI_LUU_MODEL.ipynb.
Do not change without retraining and verifying predictions.
"""

import numpy as np
import pandas as pd

RAW_FEATURES = ["vendor_id","pickup_month","pickup_dayofweek","pickup_hour",
    "pickup_minute_bucket","passenger_count","pickup_lon_approx","pickup_lat_approx",
    "dropoff_lon_approx","dropoff_lat_approx","straight_line_km_approx","store_and_fwd_flag"]
CATS_BASE = ["vendor_id","pickup_month","pickup_dayofweek","pickup_minute_bucket",
             "store_and_fwd_flag","pickup_cell","dropoff_cell","route_cell"]
RANGES = {"vendor_id":(1,2),"pickup_month":(1,12),"pickup_dayofweek":(0,6),
    "pickup_hour":(0,23),"pickup_minute_bucket":(0,45),"passenger_count":(1,6),
    "pickup_lon_approx":(-180,180),"dropoff_lon_approx":(-180,180),
    "pickup_lat_approx":(-90,90),"dropoff_lat_approx":(-90,90),
    "straight_line_km_approx":(0,float("inf"))}
INTEGERS = {"vendor_id","pickup_month","pickup_dayofweek","pickup_hour","pickup_minute_bucket","passenger_count"}

def clean_inputs(raw):
    missing = set(RAW_FEATURES)-set(raw.columns)
    if missing: raise ValueError(f"Thiếu trường đầu vào: {sorted(missing)}")
    x = raw[RAW_FEATURES].copy()
    for c, (lo,hi) in RANGES.items():
        v=pd.to_numeric(x[c],errors="coerce").astype(float)
        bad=~np.isfinite(v)|v.lt(lo)|v.gt(hi)
        if c in INTEGERS: bad|=v.notna() & v.mod(1).ne(0)
        if c=="pickup_minute_bucket": bad|=~v.isin([0,15,30,45])
        x[c]=v.mask(bad)
    f=x["store_and_fwd_flag"].astype("string").str.upper().str.strip()
    x["store_and_fwd_flag"]=f.where(f.isin(["Y","N"]))
    return x

def make_features(raw, mode="base"):
    x=clean_inputs(raw)
    lon1,lat1=x.pickup_lon_approx,x.pickup_lat_approx
    lon2,lat2=x.dropoff_lon_approx,x.dropoff_lat_approx
    dl,da=lon2-lon1,lat2-lat1
    dx=dl*111.195*np.cos(np.deg2rad((lat1+lat2)/2))
    dy=da*111.195
    a=np.sin(np.deg2rad(da)/2)**2+np.cos(np.deg2rad(lat1))*np.cos(np.deg2rad(lat2))*np.sin(np.deg2rad(dl)/2)**2
    hav=2*6371.0088*np.arcsin(np.sqrt(np.clip(a,0,1)))
    minute=x.pickup_hour*60+x.pickup_minute_bucket
    x["delta_lon"]=dl;x["delta_lat"]=da
    x["abs_delta_lon"]=np.abs(dl);x["abs_delta_lat"]=np.abs(da)
    x["midpoint_lon"]=(lon1+lon2)/2;x["midpoint_lat"]=(lat1+lat2)/2
    x["delta_x_km"]=dx;x["delta_y_km"]=dy
    x["haversine_km"]=hav;x["manhattan_km"]=np.abs(dx)+np.abs(dy)
    x["distance_rounding_gap"]=x.straight_line_km_approx-hav
    x["distance_ratio"]=x.straight_line_km_approx/(hav+0.25)
    x["log_distance"]=np.log1p(x.straight_line_km_approx.clip(lower=0))
    x["distance_zero"]=(x.straight_line_km_approx==0).astype(int)
    x["minute_of_day"]=minute
    x["hour_sin"]=np.sin(2*np.pi*minute/1440);x["hour_cos"]=np.cos(2*np.pi*minute/1440)
    x["weekday_sin"]=np.sin(2*np.pi*x.pickup_dayofweek/7)
    x["weekday_cos"]=np.cos(2*np.pi*x.pickup_dayofweek/7)
    angle=np.arctan2(dx,dy)
    x["direction_sin"]=np.sin(angle);x["direction_cos"]=np.cos(angle)
    def cell(lon,lat):
        return ((lon*100).round().fillna(-99999).astype("int64").astype(str)
                +"_"+(lat*100).round().fillna(-99999).astype("int64").astype(str))
    x["pickup_cell"]=cell(lon1,lat1)
    x["dropoff_cell"]=cell(lon2,lat2)
    x["route_cell"]=x.pickup_cell+":"+x.dropoff_cell
    x["same_cell"]=(x.pickup_cell==x.dropoff_cell).astype(int)
    cats=CATS_BASE.copy()
    if mode=="compact":
        w=x.pickup_dayofweek
        x["weekend"]=(w>=5).astype(float)
        h=x.pickup_hour.fillna(-1).astype(int)
        w1=w.fillna(-1).astype(int)
        x["time_of_week"]=(w1*24+h).astype(str)
        cats.append("time_of_week")
        for deg in [35,45]:
            t=np.deg2rad(deg)
            u=dx*np.cos(t)+dy*np.sin(t)
            v=-dx*np.sin(t)+dy*np.cos(t)
            x[f"rotated_manhattan_{deg}"]=np.abs(u)+np.abs(v)
        x["log_distance_hour_sin"]=x.log_distance*x.hour_sin
        x["log_distance_hour_cos"]=x.log_distance*x.hour_cos
        x["log_distance_weekend"]=x.log_distance*x.weekend
    elif mode!="base": raise ValueError("mode phải là base hoặc compact")
    for c in cats:
        if c in RAW_FEATURES and c!="store_and_fwd_flag":
            x[c]=x[c].astype("Int64").astype("string")
        x[c]=x[c].fillna("MISSING").astype(str)
    for c in x.columns:
        if c not in cats:
            x[c]=pd.to_numeric(x[c],errors="coerce").replace([np.inf,-np.inf],np.nan).astype("float32")
    assert "trip_id" not in x and "trip_duration" not in x
    return x,cats

def fit_spec(x,cats):
    nums=[c for c in x.columns if c not in cats]
    med=x[nums].median().fillna(0)
    levels={c:sorted(set(x[c].astype(str))|{"MISSING","__UNKNOWN__"}) for c in cats}
    return {"columns":list(x.columns),"cats":cats,"nums":nums,
            "med":{c:float(med[c]) for c in nums},"levels":levels}

def transform(x,spec,for_catboost=False):
    r=x[spec["columns"]].copy()
    for c in spec["nums"]:
        r[c]=r[c].fillna(spec["med"][c]).astype("float32")
    for c in spec["cats"]:
        v=r[c].fillna("MISSING").astype(str)
        v=v.where(v.isin(spec["levels"][c]),"__UNKNOWN__")
        r[c]=v if for_catboost else pd.Categorical(v,categories=spec["levels"][c])
    return r



def from_log(v):
    return np.maximum(np.expm1(np.asarray(v, dtype=float)), 1.0)
