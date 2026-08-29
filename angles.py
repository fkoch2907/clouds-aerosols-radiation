import numpy as np
import matplotlib.pyplot as plt
import pvlib
import pandas as pd
#%%
sp_sun_path = "microtops_sunmeasurements.xlsx"
sp_cloud_path = "microtops_cloudmeasurements.xlsx"

sp_sun = pd.read_excel(sp_sun_path)
sp_cloud = pd.read_excel(sp_cloud_path)

#%% Cloud brightness = irradiance

# -999 durch NaN ersetzen
for wl in [380, 500, 870, 936, 1020]:
    sp_cloud[f"SIG{wl}"] = np.where(
        sp_cloud[f"SIG{wl}"] == -999,
        np.nan,
        sp_cloud[f"SIG{wl}"]
    )

# Irradiance berechnen
sp_cloud["irr_cloud_380"] = sp_cloud["C1"] * sp_cloud["SIG380"]
sp_cloud["irr_cloud_500"] = sp_cloud["C2"] * sp_cloud["SIG500"]
sp_cloud["irr_cloud_870"] = sp_cloud["C3"] * sp_cloud["SIG870"]
sp_cloud["irr_cloud_936"] = sp_cloud["C4"] * sp_cloud["SIG936"]
sp_cloud["irr_cloud_1020"] = sp_cloud["C5"] * sp_cloud["SIG1020"]


#%% Position of the sun

latitude = sp_sun["LATITUDE"]
longitude = sp_sun["LONGITUDE"]

sp_sun_datetime = pd.to_datetime(
    sp_sun["DATE"].astype(str) + " " + sp_sun["TIME"].astype(str)
)

sp_cloud_datetime = pd.to_datetime(
    sp_cloud["DATE"].astype(str) + " " + sp_cloud["TIME"].astype(str)
)

time = pd.DatetimeIndex(
    sp_sun_datetime,
    tz="UTC"
)

position = pvlib.solarposition.get_solarposition(
    time,
    latitude,
    longitude
)


#%% Defining vectors towards the sun and the cloud

def vector(elevation_deg, azimuth_deg):

    elevation = np.radians(elevation_deg)
    azimuth = np.radians(azimuth_deg)

    x = np.cos(elevation) * np.sin(azimuth)  # Osten
    y = np.cos(elevation) * np.cos(azimuth)  # Norden
    z = np.sin(elevation)                    # oben

    return np.array([x, y, z])


cloud_vector = -vector(
    sp_cloud["CZA"],
    sp_cloud["CAZ"]
)

sun_vector = -vector(
    position["elevation"],
    position["azimuth"]
)


#%% Calculate angle between vectors

dot_products = np.sum(
    sun_vector * cloud_vector,
    axis=0
)

sun_norm = np.linalg.norm(
    sun_vector,
    axis=0
)

cloud_norm = np.linalg.norm(
    cloud_vector,
    axis=0
)

cos_theta = dot_products / (
    sun_norm * cloud_norm
)

theta = np.degrees(
    np.arccos(
        np.clip(cos_theta, -1.0, 1.0)
    )
)


#%% Plot theta against brightness
cloud_colors = {
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


wavelengths = [380, 500, 870, 936, 1020]

fig, ax = plt.subplots(figsize = (10, 4))

for cloud_type, group in sp_cloud.groupby("Cum"):

    idx = group.index

    color = cloud_colors.get(cloud_type, "grey")

    for wl in wavelengths:

        ax.scatter(
            theta[idx],
            sp_cloud.loc[idx, f"irr_cloud_{wl}"],
            s=10,
            alpha=0.4,
            color=color,
            label=None
        )

    ax.scatter(
        [],
        [],
        color=cloud_colors.get(cloud_type, "gray"),
        label=cloud_labels.get(cloud_type, cloud_type)
    )


ax.set_xlabel("Angle Between Sun and Cloud (degrees)", fontsize = 12)
ax.set_ylabel(r"Measured Cloud Irradiance ($W/m^2$)", fontsize = 12)

ax.set_ylim(0, 0.01)
ax.set_xlim(0, 180)

ax.set_title("Cloud Irradiance vs. Sun-Cloud Angle", fontsize = 16)
plt.tight_layout()
ax.legend(title="Cloud Type")
plt.savefig("Cloud_irr_angle_final.png")
plt.show()

