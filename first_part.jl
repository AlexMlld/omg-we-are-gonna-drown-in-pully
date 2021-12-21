
 begin
    #Imports required packages
    using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using CSV, DataFrames, MLJ, NearestNeighborModels, MLCourse, Random, Plots,MLJLinearModels, OpenML, Flux

    #Imports training, test and a sample submission data
    training = CSV.read(joinpath(@__DIR__, "Data", "trainingdata.csv"),DataFrame)
    testing=CSV.read(joinpath(@__DIR__, "Data", "testdata.csv"),DataFrame)
end


begin
    #Drops the Missing Data
    training1=dropmissing(training)
end


#KNNRegressor: ATTENTION: MAY NOT BE THE SAME PARAMETERS FROM SUBMISSION 1
begin
    model1 = KNNRegressor()
    self_tuning_kNN = TunedModel(model = model1,
                                   resampling = CV(nfolds =5),
                                   tuning = Grid(),
                                   range = range(model1, :K, values = 1:20),
                                   measure = rmse)
    self_tuning_kNN_mach = machine(self_tuning_kNN,
                               select(training1, Not(:precipitation_nextday)),
                               training1.precipitation_nextday) |> fit!
end



#RidgeRegressor: ATTENTION: MAY NOT BE THE SAME PARAMETERS FROM SUBMISSION 2
begin
    model2 = RidgeRegressor()
    self_tuning_ridge = TunedModel(model = model2,
	                         resampling = CV(nfolds = 10),
	                         tuning = Grid(goal = 200),
	                         range = range(model2, :lambda,
									       scale = :log,
									       lower = 1e-8, upper = 1e-6),
	                         measure = rmse)
    
    self_tuning_ridge_mach = machine(self_tuning_ridge,
                               select(training1, Not(:precipitation_nextday)),
                               training1.precipitation_nextday) |> fit!
end



#LinearRegressor:
begin
    model4 = LinearRegressor()
    
    
    linear_mach = machine(model4,
                               select(training1, Not(:precipitation_nextday)),
                               training1.precipitation_nextday) |> fit!
end




#Tree Based Regressor: Submission 4
begin
    using MLJXGBoostInterface,MLJDecisionTreeInterface
    weather_input = MLJ.transform(machine(OneHotEncoder(drop_last = true),
	                               select(training1, Not(:precipitation_nextday))) |> fit!,
                                   select(training1, Not(:precipitation_nextday)))

	xgb = XGBoostRegressor()
    self_tuning_xgb_mach = machine(TunedModel(model = xgb,
                            resampling = CV(nfolds = 10),
                            tuning = Grid(goal=25),
                            range = [range(xgb, :eta,
                                           lower = 1e-2, upper = .1, scale = :log),
                                     range(xgb, :num_round, lower = 50, upper = 500),
                                     range(xgb, :max_depth, lower = 2, upper = 6)]),
                  weather_input, training1.precipitation_nextday) |> fit!
                
end



#Function that makes sure the value is a probability (between 0 and 1)
#If the value in the array is bigger than 1, it becomes 1
#If the value in the array is smaller than 0, it becomes 0
#The corrected array is then returned
begin
    function enemy_of_out_of_bounds(prediction)
        for k in enumerate(prediction)
            if k[2]>=1
                prediction[k[1]]=1
            elseif k[2]<=0
                prediction[k[1]]=0
            end
        end
        prediction
    end
end


#Outputs a DataFrame of the Training RMSEs of Multiple Models
begin
    
    #Predicts the dataset with all values between 0 and 1
    pred_kNN= enemy_of_out_of_bounds(predict(self_tuning_kNN_mach, select(training1, Not(:precipitation_nextday))))
    pred_ridge= enemy_of_out_of_bounds(predict(self_tuning_ridge_mach, select(training1, Not(:precipitation_nextday))))
    pred_poly= enemy_of_out_of_bounds(predict(self_tuning_poly_mach, select(training1, Not(:precipitation_nextday))))
    pred_linear= enemy_of_out_of_bounds(predict(linear_mach, select(training1, Not(:precipitation_nextday))))
    pred_xgb= enemy_of_out_of_bounds(predict(self_tuning_xgb_mach, select(training1, Not(:precipitation_nextday))))

    #Calculates the RMSEs
    rmse_kNN=rmse(pred_kNN, training1.precipitation_nextday)
    rmse_ridge=rmse(pred_ridge, training1.precipitation_nextday)
    rmse_poly=rmse(pred_poly, training1.precipitation_nextday)
    rmse_linear=rmse(pred_linear, training1.precipitation_nextday)
    rmse_pred_xgb=rmse(pred_xgb, training1.precipitation_nextday)

    #Creates DataFrame
    training_errors=DataFrame(Models= ["kNN Regression","Ridge Regression","PolynomialRegression", "LinearRegression", "XGB"],
                                                                         Training_RMSE=[rmse_kNN,rmse_ridge,rmse_poly,rmse_linear, rmse_pred_xgb])
end


#Reports of the Various Models
report(self_tuning_kNN_mach)
report(self_tuning_ridge_mach)
report(self_tuning_poly_mach)




#Constructs DataFrames of test data predictions
begin

    #Predictions based on test data for KNNRegressor
    prediction1=enemy_of_out_of_bounds(predict(self_tuning_kNN_mach, testing))
    submission1_kNN=DataFrame(id=[i[1] for i in enumerate(prediction1)],precipitation_nextday=prediction1)


    #Predictions based on test data for Ridge Regressor
    prediction2=enemy_of_out_of_bounds(predict(self_tuning_ridge_mach, testing))
    submission2_ridge=DataFrame(id=[i[1] for i in enumerate(prediction2)],precipitation_nextday=prediction2)


    #Predictions based on test data for Linear Regressor
    prediction3=enemy_of_out_of_bounds(predict(linear_mach, testing))
    submission3_linear=DataFrame(id=[i[1] for i in enumerate(prediction3)],precipitation_nextday=prediction3)

    #Predictions based on test data for XGB    
    prediction4=enemy_of_out_of_bounds(predict(self_tuning_xgb_mach, testing))
    submission4_xgb=DataFrame(id=[i[1] for i in enumerate(prediction4)],precipitation_nextday=prediction4)
end


#Writes Submission Data from the previously constructed test data predictions
begin
    CSV.write(joinpath(@__DIR__, "Submission_Data", "submission1_kNN.csv"), submission1_kNN)
    CSV.write(joinpath(@__DIR__, "Submission_Data", "submission2_ridge.csv"), submission2_ridge)
    CSV.write(joinpath(@__DIR__, "Submission_Data", "submission3_linear.csv"), submission3_linear)
    CSV.write(joinpath(@__DIR__, "Submission_Data", "submission4_xgb.csv"), submission4_xgb)
end