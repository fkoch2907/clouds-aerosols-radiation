import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#%%
sp_sun_path = "microtops_sunmeasurements.xlsx"
sp_cloud_path = "microtops_cloudmeasurements.xlsx"
sp_contrail_path = "LEXDATA/contrail_measurements.xlsx"

sp_sun = pd.read_excel(sp_sun_path)
sp_cloud = pd.read_excel(sp_cloud_path)
sp_contrail = pd.read_excel(sp_contrail_path)

sp_sun_group = sp_sun["Group"]
sp_cloud_group = sp_cloud["Group"]
sp_contrail_group = sp_contrail["Group"]

sp_sun_datetime = pd.to_datetime(sp_sun['DATE'].astype(str) + ' ' + sp_sun['TIME'].astype(str))
sp_cloud_datetime = pd.to_datetime(sp_cloud['DATE'].astype(str) + ' ' + sp_cloud['TIME'].astype(str))
sp_contrail_datetime = pd.to_datetime(sp_contrail['DATE'] + ' ' + sp_contrail['TIME'])

sp_sun_sig380 = sp_sun["SIG380"]
sp_cloud_sig380 = sp_cloud["SIG380"]
sp_contrail_sig380 = sp_contrail["SIG380"]
sp_sun_sig380 = np.where(sp_sun_sig380 == -999, np.nan, sp_sun_sig380)
sp_cloud_sig380 = np.where(sp_cloud_sig380 == -999, np.nan, sp_cloud_sig380)
sp_contrail_sig380 = np.where(sp_contrail_sig380 == -999, np.nan, sp_contrail_sig380)

sp_sun_sig500 = sp_sun["SIG500"]
sp_cloud_sig500 = sp_cloud["SIG500"]
sp_contrail_sig500 = sp_contrail["SIG500"]
sp_sun_sig500 = np.where(sp_sun_sig500 == -999, np.nan, sp_sun_sig500)
sp_cloud_sig500 = np.where(sp_cloud_sig500 == -999, np.nan, sp_cloud_sig500)
sp_contrail_sig500 = np.where(sp_contrail_sig500 == -999, np.nan, sp_contrail_sig500)

sp_sun_sig870 = sp_sun["SIG870"]
sp_cloud_sig870 = sp_cloud["SIG870"]
sp_contrail_sig870 = sp_contrail["SIG870"]
sp_sun_sig870 = np.where(sp_sun_sig870 == -999, np.nan, sp_sun_sig870)
sp_cloud_sig870 = np.where(sp_cloud_sig870 == -999, np.nan, sp_cloud_sig870)
sp_contrail_sig870 = np.where(sp_contrail_sig870 == -999, np.nan, sp_contrail_sig870)

sp_sun_sig936 = sp_sun["SIG936"]
sp_cloud_sig936 = sp_cloud["SIG936"]
sp_contrail_sig936 = sp_contrail["SIG936"]
sp_sun_sig936 = np.where(sp_sun_sig936 == -999, np.nan, sp_sun_sig936)
sp_cloud_sig936 = np.where(sp_cloud_sig936 == -999, np.nan, sp_cloud_sig936)
sp_contrail_sig936 = np.where(sp_contrail_sig936 == -999, np.nan, sp_contrail_sig936)

sp_sun_sig1020 = sp_sun["SIG1020"]
sp_cloud_sig1020 = sp_cloud["SIG1020"]
sp_contrail_sig1020 = sp_contrail["SIG1020"]
sp_sun_sig1020 = np.where(sp_sun_sig1020 == -999, np.nan, sp_sun_sig1020)
sp_cloud_sig1020 = np.where(sp_cloud_sig1020 == -999, np.nan, sp_cloud_sig1020)
sp_contrail_sig1020 = np.where(sp_contrail_sig1020 == -999, np.nan, sp_contrail_sig1020)

#%% calculating the irradiance:
sp_cloud["irr_cloud_380"] = sp_cloud["C1"] * sp_cloud["SIG380"]
sp_cloud["irr_cloud_500"] = sp_cloud["C2"] * sp_cloud["SIG500"]
sp_cloud["irr_cloud_870"] = sp_cloud["C3"] * sp_cloud["SIG870"]
sp_cloud["irr_cloud_936"] = sp_cloud["C4"] * sp_cloud["SIG936"]
sp_cloud["irr_cloud_1020"] = sp_cloud["C5"] * sp_cloud["SIG1020"]

sp_sun["irr_sun_380"] = sp_sun["C1"] * sp_sun["SIG380"]
sp_sun["irr_sun_500"] = sp_sun["C2"] * sp_sun["SIG500"]
sp_sun["irr_sun_870"] = sp_sun["C3"] * sp_sun["SIG870"]
sp_sun["irr_sun_936"] = sp_sun["C4"] * sp_sun["SIG936"]
sp_sun["irr_sun_1020"] = sp_sun["C5"] * sp_sun["SIG1020"]

#%%
fig, ax = plt.subplots()
ax.scatter(sp_cloud_group, sp_cloud["irr_cloud_380"], label = "380", alpha = 0.5)
ax.scatter(sp_cloud_group, sp_cloud["irr_cloud_500"], label = "500", alpha = 0.5)
ax.scatter(sp_cloud_group, sp_cloud["irr_cloud_870"], label = "870", alpha = 0.5)
ax.scatter(sp_cloud_group, sp_cloud["irr_cloud_936"], label = "936", alpha = 0.5)
ax.scatter(sp_cloud_group, sp_cloud["irr_cloud_1020"], label = "1020", alpha = 0.5)
ax.set_xlabel("measurement")
ax.set_ylabel(r"Irradiance of cloud [$W/m^2$]")
ax.set_ylim(0, 0.01)
plt.legend()
plt.savefig("irr_cloud_measured.png")
plt.show()

#%%
reflected_by_cloud380 = sp_cloud["irr_cloud_380"] /(0.5 - 0.5 * (np.cos(np.deg2rad(1.25)))**2)
reflected_by_cloud500 = sp_cloud["irr_cloud_500"] /(0.5 - 0.5 * (np.cos(np.deg2rad(1.25)))**2)
reflected_by_cloud870 = sp_cloud["irr_cloud_870"] /(0.5 - 0.5 * (np.cos(np.deg2rad(1.25)))**2)
reflected_by_cloud936 = sp_cloud["irr_cloud_936"] /(0.5 - 0.5 * (np.cos(np.deg2rad(1.25)))**2)
reflected_by_cloud1020 = sp_cloud["irr_cloud_1020"] /(0.5 - 0.5 * (np.cos(np.deg2rad(1.25)))**2)

fig, ax = plt.subplots()
ax.scatter(sp_sun["irr_sun_380"], reflected_by_cloud380, label = "380")
ax.scatter(sp_sun["irr_sun_500"], reflected_by_cloud500, label = "500")
ax.scatter(sp_sun["irr_sun_870"], reflected_by_cloud870, label = "870")
ax.scatter(sp_sun["irr_sun_936"], reflected_by_cloud936, label = "936")
ax.scatter(sp_sun["irr_sun_1020"], reflected_by_cloud1020, label = "1020")
ax.set_xlabel("irradiance of sun")
ax.set_ylabel("irradiance of cloud")
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
plt.legend()
plt.savefig("irr_cloud_sun_comparison.png")
plt.show()

#%%
fig, ax = plt.subplots()
ax.scatter(sp_sun_group, sp_sun["irr_sun_380"], color = "orange", alpha = 0.5, label = "sun")
ax.scatter(sp_cloud_group, reflected_by_cloud380, color = "blue", alpha = 0.5, label = "cloud")
ax.set_ylim(0, 12)
ax.set_title("380")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.scatter(sp_sun_group, sp_sun["irr_sun_500"], color = "orange", alpha = 0.5, label = "sun")
ax.scatter(sp_cloud_group, reflected_by_cloud500, color = "blue", alpha = 0.5, label = "cloud")
ax.set_ylim(0, 40)
ax.set_title("500")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.scatter(sp_sun_group, sp_sun["irr_sun_870"], color = "orange", alpha = 0.5, label = "sun")
ax.scatter(sp_cloud_group, reflected_by_cloud870, color = "blue", alpha = 0.5, label = "cloud")
ax.set_ylim(0, 12)
ax.set_title("870")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.scatter(sp_sun_group, sp_sun["irr_sun_936"], color = "orange", alpha = 0.5, label = "sun")
ax.scatter(sp_cloud_group, reflected_by_cloud936, color = "blue", alpha = 0.5, label = "cloud")
ax.set_ylim(0, 6)
ax.set_title("936")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.scatter(sp_sun_group, sp_sun["irr_sun_1020"], color = "orange", alpha = 0.5, label = "sun")
ax.scatter(sp_cloud_group, reflected_by_cloud1020, color = "blue", alpha = 0.5, label = "cloud")
ax.set_ylim(0, 12)
ax.set_title("1020")
plt.legend()
plt.show()

#%%
colors = {
    "Cir": "blue",
    "Cum": "green",
    "Altocum": "red",
    "Altostra": "yellow",
    "Stra": "orange"
}
cloud_labels = {
    "Cum": "Cumulus",
    "Stra": "Stratus",
    "Cir": "Cirrus",
    "Altocum": "Altocumulus",
    "Altostra": "Altostratus"
}

#%%

wavelengths = [380, 500, 870, 936, 1020]

#%%

irr_sun = {
    380: sp_sun["irr_sun_380"],
    500: sp_sun["irr_sun_500"],
    870: sp_sun["irr_sun_870"],
    936: sp_sun["irr_sun_936"],
    1020: sp_sun["irr_sun_1020"]
}

reflected_by_cloud = {
    380: reflected_by_cloud380,
    500: reflected_by_cloud500,
    870: reflected_by_cloud870,
    936: reflected_by_cloud936,
    1020: reflected_by_cloud1020
}

fig, ax = plt.subplots()

for cloud_type, group in sp_cloud.groupby("Cum"):

    color = colors.get(cloud_type, "gray")

    # Index der Messungen dieses Wolkentyps
    idx = group.index

    for wl in wavelengths:
        ax.scatter(
            irr_sun[wl].loc[idx],
            reflected_by_cloud[wl].loc[idx],
            color=color,
            s=10,
            alpha=0.5
        )

    ax.scatter(
        [],
        [],
        color=color,
        label=cloud_type
    )

ax.set_xlim(0, 12)
ax.set_ylim(0, 12)

ax.set_xlabel("Irradiance sun")
ax.set_ylabel("Reflected irradiance")

ax.legend(title="Cloud type")
plt.plot([0, 12], [0, 12], color='grey', linestyle='--')
plt.show()

#%%
fig, ax = plt.subplots(figsize = (10,4))

for cloud_type, group in sp_cloud.groupby("Cum"):
    idx = group.index
    for i, wl in enumerate(wavelengths):
        ax.scatter(
            idx,
            sp_cloud.loc[idx, f"irr_cloud_{wl}"],
            s=10,
            alpha=0.5,
            color=colors.get(cloud_type, "gray"),
            label=cloud_labels.get(cloud_type, cloud_type) if i == 0 else None
        )



ax.set_ylim(0, 0.01)

ax.set_xlabel("Measurement", fontsize = 12)
ax.set_ylabel(r"Cloud Irradiance [$W/m^2$]", fontsize = 12)
ax.set_title("Cloud Irradiance Measured by Sunphotometer", fontsize = 16)
ax.legend(title="Cloud Type")
plt.tight_layout()
plt.savefig("Cloud_irr_withtype_measured.png")
plt.show()


#%%
fig, ax = plt.subplots(figsize = (10,4))

for cloud_type, group in sp_cloud.groupby("Cum"):
    idx = group.index
    for i, wl in enumerate(wavelengths):
        ax.scatter(
            sp_sun.loc[idx, f"AOT{wl}"],
            sp_cloud.loc[idx, f"irr_cloud_{wl}"],
            s=10,
            alpha=0.5,
            color=colors.get(cloud_type, "gray"),
            label=cloud_labels.get(cloud_type, cloud_type) if i == 0 else None
        )



ax.set_ylim(0, 0.01)

ax.set_xlabel("AOT", fontsize = 12)
ax.set_ylabel(r"Cloud Irradiance [$W/m^2$]", fontsize = 12)
ax.set_title("Cloud Irradiance vs. AOT", fontsize = 16)
ax.legend(title="Cloud Type")
ax.set_xlim(0, 0.4)
plt.tight_layout()
plt.savefig("Cloud_irr_AOT_measured.png")
plt.show()