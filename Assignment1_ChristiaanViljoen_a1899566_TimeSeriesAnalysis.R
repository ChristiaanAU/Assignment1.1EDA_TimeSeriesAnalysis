#1. Save the data as a CSV file
#2. Make the working directory the same place as the data 
#3. read in using below code
#4. transpose it and make it a data frame

# Import CSV and make first column row names
df <- read.csv("BirthsAndFertilityRatesAnnualCSV.csv", 
               header = TRUE,        # First row is column names
               row.names = 1,        # Use first column as row names
               stringsAsFactors = FALSE,
               check.names = FALSE
)

df <- as.data.frame(t(df)) # Looks strange but simply to make the first column the subject


#We only need Total Live Births (TLB) and Total Fertility Rate (TFR) from 1960 to 2024.

TLB <- as.numeric(df$`Total Live-Births`)
TFR <- as.numeric(df$`Total Fertility Rate (TFR)`)

#Although the years have been cut off and it is a simple 1 to length(), it is not confusing
#and should still be simple enough to work with

plot(TFR)
plot(TLB)

#The above has shown me that I literally need to reverse the order of the values so yeah

TLB <- log(TLB) |> rev() 
TFR <- log(TFR) |> rev()

plot(TLB)
plot(TFR)

#just make sure that it is the correct way but otherwise looks good now can work with it
#What is the first thing we shall do?
#Perhaps make sure that there is no NA in the data

TLB |> is.na() |> any()
TFR |> is.na() |> any()

#Both are false so no need to clean the data of NA... now we can work with it


library(tsibble)

years <- 1960:2025

TFR <- data.frame(year = years, TFR = TFR)
TLB <- data.frame(year = years, TLB = TLB)

TFR <- as_tsibble(TFR, index = year)
TLB <- as_tsibble(TLB, index = year)

plot(TFR)
plot(TLB)

#Looking at the plot of TFR, it is clear to see that it has a downward trend.
#It does not seem, however, to have heteroskedasticity, cycles, or seasonality.
#The TLB looks a bit more complex, however. Again, no heteroskedasticity and 
#we expect no seasonality. However, it does look possible that there could be 
#some cycles in the data, as it has long-term downard trend but no clear
#medium term trend(s).



#This concludes the data cleaning of TFR and TLB.


#To understand what would be done in this analysis, we are trying
#to create models where the residuals follow white noise. This means that they have to 
#following the criterion: 

#1. uncorrelated random variables
#2. zero mean
#3. the same variance throughout

#For the first criterion, we need the portmanteau tests to have test 
#statistics that are as low as possible and, although not 100% correct,
#preferably the ACF to decay as fast as possible (can have stationary
#and slow decay but as mentioned this is not preferable).
#For the mean, simply a visual plot that shows no clear change in
#the direction of the data is all that is needed.
#For the variance, again, a simple visual plot that shows no clear
#spread or shrinkage in the data is also good enough. 

######################### TFR model 1 ##########################

library(dplyr)
library(tsibble)
library(feasts)
library(fable)

index_var(TFR)
frequency(TFR)
common_periods(TFR)
key_vars(TFR)
measured_vars(TFR)
scan_gaps(TFR)

#As can be seen from the above analysis, the data is as we want it
#structurally. Also, the frequency of the data is 1, and thus it 
#is not seasonal (as expected given it is annual data).

TFR |> pull(year) |> range()

#From the above, the range in the years from beginning to the end
#is correct, and thus we are ready to start to analyse the data.

library(ggtime)

TFR |> gg_tsdisplay(y = TFR, plot_type = "spectrum")
TFR |> gg_tsdisplay(y = TFR, plot_type = "partial")

#From the above, it is clear to see that there is no inherent 
#frequency in the model and thus no seasonality. The acf 
#shows a slow decay, suggesting a strong autoregressive or 
#non-stationarity structure; differencing is justified. 

TFR <- TFR |>
  mutate(D1 = difference(TFR))

TFR |> pull(D1) |> plot()

TFR_D1 <- TFR |> filter(!is.na(D1))

TFR |> features(TFR, portmanteau_tests)
TFR_D1 |> features(D1, portmanteau_tests)

#The above results, which is shown here : 
#> TFR |> features(TFR, portmanteau_tests)
# A tibble: 1 × 4
#lb_stat lb_pvalue bp_stat bp_pvalue
#<dbl>     <dbl>   <dbl>     <dbl>
#  1    59.4  1.30e-14    56.8  4.92e-14
#> TFR |> features(D1, portmanteau_tests)
# A tibble: 1 × 4
#lb_stat lb_pvalue bp_stat bp_pvalue
#<dbl>     <dbl>   <dbl>     <dbl>
#  1  0.0182     0.893  0.0174     0.895

#Shows that using the differenced data is much better than using
#the data as it is. The portmanteau test suggest that the data 
#has no remaining linear correlation. 

TFR |> filter(!is.na(D1)) |> ACF(D1, lag_max = 50) |> autoplot()
TFR |> filter(!is.na(D1)) |> PACF(D1, lag_max = 50) |> autoplot() 

#As can be seen from the ACF and PACF, altough it does look like big 
#jumps, this is expected as we expect that 95% of the spikes of the 
#acf should fall between +- 1.96/sqrt(n) (TEXTBOOK, p16), and only 4 of the 
#spikes went out of the interval (we expect 65*0.05 = 3.25). Thus,
#we can safely say that there truly is not autocorrelation left 
#in the structure. Now, we need to look for constant mean and 
#constant variance. 

TFR_D1 |> pull(D1) |> plot() 

#Looking at the data, it seems as if the variance and the mean
#stays constant, as there is no clear spread or change in the data. 
#Thus, the time series has been made stationary through a simple
#one time differencing. Now, we can create models it so that the 
#residuals are white noise.

TFR_D1_MODEL <- TFR |> 
  model(I_1 = ARIMA(TFR ~ pdq(0,1,0)))

mean_res <- mean(augment(TFR_D1_MODEL)$.resid, na.rm = TRUE)
sd_res <- sd(augment(TFR_D1_MODEL)$.resid, na.rm = TRUE)
#the mean is basically zero and the standard deviation is 0.062

TFR_D1_MODEL |> gg_tsresiduals(plot_type = "spectrum")
TFR_D1_MODEL |> gg_tsresiduals(plot_type = "partial")

#From the abovce, we can confidently say that an ARIMA structure of 
#difference of 1 accurately describes the data. This implies that 
#time series is a random walk model i.e. Xt = X(t-1) + ei, where 
#ei ~ WN(0, 0.062^2).

tidy(TFR_D1_MODEL)
glance(TFR_D1_MODEL)

#Note that the constant of -0.0291 is very significant, and thus a 
#model of X(t) - X(t-1) = -0.0291 + ei is the correct equation.

########################## TFR model 2 #############################

#The model of ARIMA(0,1,0), albeit being simple and does explain
#the data in this context, tends to be too simplistic. We need either
#a better ARIMA model or something that captures the fact that it is 
#a downard trend with no seasonality, such as a STL decomposition. 

#First, lets try making ARIMA more complex as the first model is already
#a ARIMA model. 

TFR_model2 <- TFR |>
  model(DAR1 = ARIMA(TFR ~ pdq(1,1,0)))

TFR_model2 |> gg_tsresiduals(plot_type = "spectrum")
TFR_model2 |> gg_tsresiduals(plot_type = "partial")

tidy(TFR_model2)
glance(TFR_model2)

#As can be seen, almost nothing was gained in comparison to the 
#simple random walk. What is interisting to note here is that 
#the p-value for the AR(1) term is 0.895, making it very high
#and thus not significant. 
#Now, lets try instead to create a AR(1) model of the 
#differenced data

TFR_model3 <- TFR_D1 |> 
  model(DAR1_2 = ARIMA(D1 ~ pdq(1,0,0)))

TFR_model3 |> gg_tsresiduals(plot_type = "spectrum")
TFR_model3 |> gg_tsresiduals(plot_type = "partial")

tidy(TFR_model2)
glance(TFR_model2)

#Again, no better explanation, except that the constant is significant.

TFR_model4 <- TFR |>
  model(MA1 = ARIMA(TFR ~ pdq(0,1,1)))

TFR_model4 |> gg_tsresiduals(plot_type = "spectrum")
TFR_model4 |> gg_tsresiduals(plot_type = "partial")

tidy(TFR_model4)
glance(TFR_model4)

#Again, the ma1 term is not significant and thus an ARIMA model 
#that is more complex than a simple random walk fails to capture 
#the data. Now, lets try a STL decomposition: 

TFR_model5 <- TFR |>
  model(Ets = ETS(TFR))

TFR_model5 |> gg_tsresiduals(plot_type = "spectrum", lag_max = 50)
TFR_model5 |> gg_tsresiduals(plot_type = "partial", lag_max = 50)

Res_model5 <- resid(TFR_model5)

tidy(TFR_model5)

Res_model5 |> features(.resid, portmanteau_tests)

#As can be seen, this model is very succesful (unlike the others).
#The reason for this is that the remainders are small, have the 
#same variance, same mean, the acf looks good and the portmanteau
#test has values of a p-value about 0.9, showing that there is no
#definitive autocorrelation in the residuals. 


#Thus, two models have been found for the TFR, consisting of the
#ARIMA(0,1,0) or differencing model, and the ETS decomposition 
#model.







########################## TLB model 1 #############################
library(dplyr)
library(tsibble)
library(feasts)
library(fable)
library(ggtime)


index_var(TLB)
frequency(TLB)
common_periods(TLB)
key_vars(TLB)
measured_vars(TLB)
scan_gaps(TLB)

TLB |> pull(year) |> range()

#The data looks clean. The frequency is 1, which suggests that the 
#data does not have seasonal effects. 

TLB |> gg_tsdisplay(y = TLB, plot_type = "spectrum", lag_max = 50)
TLB |> gg_tsdisplay(y = TLB, plot_type = "partial", lag_max = 50)

#The periodogram simply decays, showing no spikes and no seasonal
#effects. Looking at the PACF, it might indicate a strong 
#autoregressive model of 1. The ACF slowly decays, indicating either
#an autoregressive model or a non-stationary process.

TLB |> features(TLB, portmanteau_tests)

#The results of the portmanteau tests shows very strong non-stationary
#data. Thus, we will difference once and see if any difference can be 
#seen.

TLB <- TLB |>
  mutate(D1 = difference(TLB))

TLB |> filter(!is.na(D1)) |> gg_tsdisplay(y = D1, plot_type = "spectrum", lag_max = 50)

TLB_model1 <- TLB |> 
  model(D1 = ARIMA(TLB ~ pdq(0,1,0)))

TLB_model1 |> gg_tsresiduals(plot_type = "spectrum", lag_max = 50)
TLB |> features(D1, portmanteau_tests)

mean_TLB_mod1 <- augment(TLB_model1) |> pull(.resid) |> mean()
sd_TLB_mod1 <- augment(TLB_model1) |> pull(.resid) |> sd()

tidy(TLB_model1)
glance(TLB_model1)

#From the above, we can see that the spectrum is small and can be 
#ignored, making the data non-seasonal and non-cyclical. The ACF
#also looks much better as it shows no clear structure in the long
#term. The portmanteau tests have an approximate p-value of 0.35,
#making the residuals of the data look clean. Also, looking at the 
#residuals on the plot, the variance looks approximately the same
#and the mean also. Thus, this data also shows a random walk model
#with x(t) - X(t-1) = -0.0111 + ei where ei ~ WN(0,0.058^2).


########################## TLB model 2 #############################


TLB_model2 <- TLB |> 
  model(D1_AR1 = ARIMA(TLB ~ pdq(1,0,0)))

TLB_model2 |> gg_tsresiduals(plot_type = "spectrum", lag_max = 50)
TLB_model2 |> gg_tsresiduals(plot_type = "partial", lag_max = 50)

tidy(TLB_model2)
glance(TLB_model2)


#From the above, it is simple to conclude that this model does work
#as well (a AR(1) model). Both the constant of 0.345 and phi of 
#0.968 are significant, the acf and spectrum look good, same mean
#and same variance. 


########################### FORECASTING #############################

###To do the forecasting, one must first split the time between
#1960 to 2012 and then use that to forecast from 2013 to 2024.

#TFR MODEL 1 : X(t) - X(t-1) = -0.0291 + ei where ei ~ WN(0, 0.062^2)
#TFR MODEL 2 : ETS model

#TLB MODEL 1 : x(t) - X(t-1) = -0.0111 + ei where ei ~ WN(0,0.058^2)
#TLB MODEL 2 : X(t) = 0.968*X(t-1) + ei + 0.345 where ei ~ WN(0, 0.00363 ^ 2)

TFR_OLD <- TFR |> filter(year <= 2012)
TFR_NEW <- TFR |> filter(year > 2012)

TFR_MODEL1_NEW <- TFR_OLD |>
  model(D1 = ARIMA(TFR ~ pdq(0,1,0)))

glance(TFR_MODEL1_NEW)
tidy(TFR_MODEL1_NEW)

TFR_MODEL1_FC <- TFR_MODEL1_NEW |> forecast(h = 12)

TFR |> 
  filter(year >= 2012) |>
  autoplot(TFR) +
  autolayer(TFR_MODEL1_FC, alpha = 0.4)

#As can be seen, although model 1 of the TFR has a constant 
#undermodelling, given that it is in the 80% of the forecast, 
#I would say that this was a successful forecast. 

TFR_MODEL2_NEW <- TFR_OLD |>
  model(Ets = ETS(TFR))

TFR_MODEL2_FC <- TFR_MODEL2_NEW |> forecast(h = 12)

TFR |> 
  filter(year >= 2012) |>
  autoplot(TFR) +
  autolayer(TFR_MODEL2_FC, alpha = 0.4)

#looking at the graph, this forecasting looks successful, altough
#as model 1 has a constant under modelling effect.

TLB_OLD <- TLB |> filter(year <= 2012)
TLB_NEW <- TLB |> filter(year > 2013)

TLB_MODEL1_NEW <- TLB_OLD |>
  model(D1 = ARIMA(TLB ~ pdq(0,1,0)))

TLB_MODEL1_FC <- TLB_MODEL1_NEW |> forecast(h = 12)

TLB |> 
  filter(year >= 2012) |>
  autoplot(TLB) +
  autolayer(TLB_MODEL1_FC, alpha = 0.4)

#THE 80% forecast was alright until it hit 2025 where it no longer
#was inside. unlike the previous two, this one is over forecasting.

TLB_MODEL2_NEW <- TLB_OLD |>
  model(A1 = ARIMA(TLB ~ pdq(1,0,0)))

TLB_MODEL2_FC <- TLB_MODEL2_NEW |> forecast(h = 12)

TLB |> 
  filter(year >= 2012) |>
  autoplot(TLB) +
  autolayer(TLB_MODEL2_FC, alpha = 0.4)

#unlike the differencing that had decent success, this one straight fails.
#it is in the 80% band until 2023, where it falls out and goes under. 

