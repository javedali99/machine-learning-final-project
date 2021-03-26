%% Obtain Storm Surge Height for ML project: Surge =  detrended Water Level - predicted tides

%% Start by detrending water level data

% change time datum format
Tmat = datevec(Time); %1950 - 2015

%239382 is the start of 1979 (first hour of jan 1, 1979)

% get (unique) years for which we have data (accounts for potential gaps in the record)
years = unique(Tmat(:,1));

% get number of years
Nyear = length(years); %66 years of data

figure
plot(Time,WaterLevel,'color',[0 0 0],'marker','o','markerfacecolor',[1 1 1],'linewidth',2)
xlabel('Time')
ylabel('Water Level (mMSL)')

% Remove the linear trend
testPara = polyfit(Time,WaterLevel,1);
testLinTrend = testPara(1)*Time+testPara(2);

% We can use regress to check for significance!
[b,bint] = regress(WaterLevel,[ones(Nyear,1) Time]);

% plot hourly water level time series
figure
plot(Time,WaterLevel,'color',[0 0 0],'marker','o','markerfacecolor',[1 1 1],'linewidth',2)
hold on
plot(Time,testLinTrend,'color',[0 0 0])
xlabel('Time')
ylabel('Water Level (mMSL)')

% get difference between last value in LinTrend and the other values
hourlyDiff = testLinTrend(end,1) - testLinTrend;
hourlydt = WaterLevel+hourlyDiff;                 %detrended

% test if new trend is zero
testPara2 = polyfit(Time,hourlydt,1);
testLinTrend2 = testPara2(1)*Time+testPara2(2);

% plot WaterLevel time series and detrended WaterLevel
figure
plot(Time,WaterLevel,'color',[0 0 0],'marker','o','markerfacecolor',[1 1 1],'linewidth',2)
hold on
plot(Time,testLinTrend,'color',[0 0 0])
plot(Time,hourlydt,'color','r','marker','o','markerfacecolor',[1 1 1],'linewidth',2)
plot(Time,testLinTrend2,'color','r')
xlabel('Time')
ylabel('Level (mMSL)')

% Extract the Predicted Tides by running T_tide package on detrended water level 
[nameuPete,fuPete,tideconPete,xoutPete] = t_tide(hourlydt(239382:end),'error','wboot'); % start 1979


