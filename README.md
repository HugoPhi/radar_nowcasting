# Chongqing University Mathematical Modeling Campus Competition Problem C

## Description

---

# Problem C: Nowcasting of Severe Convective Precipitation

China's vast territory and complex natural conditions result in various catastrophic weather types and significant regional differences. Severe convective weather, including thunderstorms, strong winds, hail, tornadoes, and short-term heavy rainfall, poses serious threats to life safety and causes economic losses [^1]. In 2022, wind and hail disasters from strong convective weather in China accounted for a significant portion of deaths, missing persons, and direct economic losses [^1]. Due to the sudden and localized nature of strong convective weather, short-term (0-12 hours) and near (0-2 hours) forecasts are often challenging in weather forecasting.

The traditional prediction of severe convective weather relies on observational data such as radar, utilizing storm identification and tracking techniques for extrapolation predictions. Recent advancements in artificial intelligence and deep learning, driven by big data and computing power, offer promising approaches for short-term and imminent prediction fields [^2]. Two main types of models based on deep learning are Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) [^3] [^4,5].

Raindrop characteristics, influenced by air resistance during falling, can be crucial for understanding precipitation. Dual polarization radar, capable of measuring reflections in both horizontal and vertical directions, provides microphysical information about precipitation particles. This information is vital for predicting severe convective weather, including the evolution state and spatial dynamic structure of convective systems [^6] [^7,8].

To better apply dual polarization radar for improved short-term and imminent forecasts of severe convective precipitation, answer the following questions:

1. **Mathematical Model for Microphysical Feature Extraction:**
   - Input: Radar observations (ZH, ZDR, KDP) from the previous hour (10 frames).
   - Output: ZH forecast from the following hour (10 frames).

2. **Quantitative Precipitation Estimation Model:**
   - Input: ZH and ZDR.
   - Output: Precipitation. (Note: The algorithm cannot use KDP variables.)

**Explanation of Terms:**

1. **Dual Polarization Radar:**
   - A new type of weather detection radar providing richer physical information.
   - Key variables: 
      - ZH (horizontal reflectivity factor)
      - ZDR (differential reflectance)
      - KDP (differential phase shift)

2. **Z-R Relationship:**
   - Empirical relationship between radar reflectivity (Z) and precipitation (R).
   - Typically expressed as 𝑅 = 𝑎 ∗ 𝑍^𝑏, where 𝑎 and 𝑏 are empirical parameters.

**Attachment Data:**
1. [NJU-CPOL dual polarization radar data](https://box.nju.edu.cn/f/16bbb37458d3443dbf9f/?dl=1)
2. [Precipitation grid data](https://box.nju.edu.cn/f/076f5aeb2ec64b87bde8/?dl=1)

**References:**
- [Zheng et al., 2010](https://doi.org/10.3969/j.issn.1001-7313.2010.07.004)
- [Chen et al., 2017](https://doi.org/10.1175/JHM-D-16-0180.1)
- [Pan et al., 2021](https://doi.org/10.1029/2021GL095302)
- [Ravuri et al., 2021](https://www.nature.com/articles/s41586-021-03813-4)
- [Zhang et al., 2023](https://www.nature.com/articles/s41586-022-04847-1)
- [Kumjian, 2013](https://journals.ametsoc.org/view/journals/jom/1/19/jom-d-13-00016_1.xml)
- [Zhao et al., 2019](https://doi.org/10.1007/s00376-019-8189-3)
- [Wen et al., 2017](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017JD026603)

[^1]: Zheng, Y., et al. (2010). Meteorology, 36(7), 33-42.
[^2]: Chen, G., et al. (2017). Journal of Hydrometeorology, 18(5), 1375-1391.
[^3]: Pan, X., et al. (2021). Geophysical Research Letters, 48(21), e2021GL095302.
[^4]: Ravuri, S., et al. (2021). Nature, 597, 672-677.
[^5]: Zhang, Y., et al. (2023). Nature, 619, 526–532.
[^6]: Kumjian, M. R. (2013). Journal of Operational Meteorology, 1(19), 226-242.
[^7]: Zhao, K., et al. (2019). Advances in Atmospheric Sciences, 36, 961-974.
[^8]: Wen, J., et al. (2017). Journal of Geophysical Research: Atmospheres, 122(15), 8033-8050.
```
--- 

## Model 


