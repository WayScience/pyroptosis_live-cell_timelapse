#!/usr/bin/env python
# coding: utf-8

# This notebook extracts a platemap of the data

# In[1]:


import pathlib

import pandas as pd

# In[ ]:


# path the the platemaps
path_to_plate_map = pathlib.Path("../../../data/processed/platemaps/").resolve()
path_to_plate_map.mkdir(parents=True, exist_ok=True)
path_to_plate_map = path_to_plate_map / "wave1_plate_map.csv"


# In[3]:


# map the wells to a given treatment for metadata purposes
dict_platemap = {
    "C04": "Media",
    "D04": "LPS 10 ug/ml",
    "E04": "LPS 1 ug/ml",
    "F04": "LPS 0.1 ug/ml",
    "G04": "LPS 1 ug/ml + ATP 2.5 mM",
    "H04": "Flagellin 10 ug/ml",
    "I04": "Flagellin 1 ug/ml",
    "J04": "Flagellin 0.1 ug/ml",
    "K04": "H2O2 500 uM",
    "L04": "H2O2 100 uM",
    "M04": "H2O2 100 nM",
    "N04": "no Hoechst",
    "C05": "DMSO CTL",
    "D05": "Thapsigargin 10 uM",
    "E05": "Thapsigargin 1 uM",
    "F05": "Thapsigargin 0.5uM",
    "G05": "LPS 1 ug/ml + Nigericin 5uM",
    "H05": "LPS 1 ug/ml + Nigericin 3uM",
    "I05": "LPS 1 ug/ml + Nigericin 1 uM",
    "J05": "LPS 1 ug/ml + Nigericin 0.5uM",
    "K05": "LPS 1 ug/ml + Nigericin 0.1 uM",
    "L05": "Ab1-42 10 uM",
    "M05": "Ab1-42 2 uM",
    "N05": "Ab1-42 0.4 uM",
    "C06": "Media",
    "D06": "LPS 10 ug/ml",
    "E06": "LPS 1 ug/ml",
    "F06": "LPS 0.1 ug/ml",
    "G06": "LPS 1 ug/ml + ATP 2.5 mM",
    "H06": "Flagellin 10 ug/ml",
    "I06": "Flagellin 1 ug/ml",
    "J06": "Flagellin 0.1 ug/ml",
    "K06": "H2O2 500 uM",
    "L06": "H2O2 100 uM",
    "M06": "H2O2 100 nM",
    "N06": "no Hoechst",
    "C07": "DMSO CTL",
    "D07": "Thapsigargin 10 uM",
    "E07": "Thapsigargin 1 uM",
    "F07": "Thapsigargin 0.5uM",
    "G07": "LPS 1 ug/ml + Nigericin 5uM",
    "H07": "LPS 1 ug/ml + Nigericin 3uM",
    "I07": "LPS 1 ug/ml + Nigericin 1 uM",
    "J07": "LPS 1 ug/ml + Nigericin 0.5uM",
    "K07": "LPS 1 ug/ml + Nigericin 0.1 uM",
    "L07": "Ab1-42 10 uM",
    "M07": "Ab1-42 2 uM",
    "N07": "Ab1-42 0.4 uM",
    "C08": "Media",
    "D08": "LPS 10 ug/ml",
    "E08": "LPS 1 ug/ml",
    "F08": "LPS 0.1 ug/ml",
    "G08": "LPS 1 ug/ml + ATP 2.5 mM",
    "H08": "Flagellin 10 ug/ml",
    "I08": "Flagellin 1 ug/ml",
    "J08": "Flagellin 0.1 ug/ml",
    "K08": "H2O2 500 uM",
    "L08": "H2O2 100 uM",
    "M08": "H2O2 100 nM",
    "N08": "no Hoechst",
    "C09": "DMSO CTL",
    "D09": "Thapsigargin 10 uM",
    "E09": "Thapsigargin 1 uM",
    "F09": "Thapsigargin 0.5uM",
    "G09": "LPS 1 ug/ml + Nigericin 5uM",
    "H09": "LPS 1 ug/ml + Nigericin 3uM",
    "I09": "LPS 1 ug/ml + Nigericin 1 uM",
    "J09": "LPS 1 ug/ml + Nigericin 0.5uM",
    "K09": "LPS 1 ug/ml + Nigericin 0.1 uM",
    "L09": "Ab1-42 10 uM",
    "M09": "Ab1-42 2 uM",
    "N09": "Ab1-42 0.4 uM",
    "C10": "Media",
    "D10": "LPS 10 ug/ml",
    "E10": "LPS 1 ug/ml",
    "F10": "LPS 0.1 ug/ml",
    "G10": "LPS 1 ug/ml + ATP 2.5 mM",
    "H10": "Flagellin 10 ug/ml",
    "I10": "Flagellin 1 ug/ml",
    "J10": "Flagellin 0.1 ug/ml",
    "K10": "H2O2 500 uM",
    "L10": "H2O2 100 uM",
    "M10": "H2O2 100 nM",
    "N10": "no Hoechst",
    "C11": "DMSO CTL",
    "D11": "Thapsigargin 10 uM",
    "E11": "Thapsigargin 1 uM",
    "F11": "Thapsigargin 0.5uM",
    "G11": "LPS 1 ug/ml + Nigericin 5uM",
    "H11": "LPS 1 ug/ml + Nigericin 3uM",
    "I11": "LPS 1 ug/ml + Nigericin 1 uM",
    "J11": "LPS 1 ug/ml + Nigericin 0.5uM",
    "K11": "LPS 1 ug/ml + Nigericin 0.1 uM",
    "L11": "Ab1-42 10 uM",
    "M11": "Ab1-42 2 uM",
    "N11": "Ab1-42 0.4 uM",
}
platemap_df = pd.DataFrame.from_dict(
    dict_platemap, orient="index", columns=["treatment"]
)
platemap_df.reset_index(inplace=True)
platemap_df.rename(columns={"index": "well"}, inplace=True)
platemap_df.head()


# In[4]:


# split the treatment column if " + "" is present
platemap_df[["treatment1", "treatment2"]] = platemap_df["treatment"].str.split(
    " \+ ", expand=True
)
platemap_df[["treatment1", "treatment1_dose"]] = platemap_df["treatment1"].str.split(
    " ", n=1, expand=True
)
platemap_df[["treatment1_dose", "treatment1_unit"]] = platemap_df[
    "treatment1_dose"
].str.split(" ", n=1, expand=True)
platemap_df[["treatment2", "treatment2_dose"]] = platemap_df["treatment2"].str.split(
    " ", n=1, expand=True
)
platemap_df[["treatment2_dose", "treatment2_unit"]] = platemap_df[
    "treatment2_dose"
].str.split(" ", n=1, expand=True)
platemap_df.head()


# In[5]:


platemap_df["treatment"].unique()
platemap_df[(platemap_df["treatment"] == "DMSO CTL")]


# In[6]:


# add FBS conitions for columns 10 and 11
platemap_df["serum"] = "FBS"
# if column 10 or 11, the NuSerum condition is added
platemap_df.loc[platemap_df["well"].str[1:] == "10", "serum"] = "NuSerum"
platemap_df.loc[platemap_df["well"].str[1:] == "11", "serum"] = "NuSerum"
# sort the df by well
platemap_df.sort_values("well", inplace=True)
# prepend Metadata to each column name
platemap_df.columns = ["Metadata_" + col for col in platemap_df.columns]
platemap_df.head()


# In[7]:


platemap_df.to_csv(path_to_plate_map, index=False)
