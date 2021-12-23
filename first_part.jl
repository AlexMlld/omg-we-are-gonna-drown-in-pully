
 begin
    #Imports required packages
    using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using CSV, DataFrames, MLJ, NearestNeighborModels, MLCourse, Random, Plots,MLJLinearModels, OpenML, Flux, MLJFlux

    #Imports training, test and a sample submission data
    training = CSV.read(joinpath(@__DIR__, "Data", "trainingdata.csv"),DataFrame)
    testing=CSV.read(joinpath(@__DIR__, "Data", "testdata.csv"),DataFrame)
end


 #Drops the Missing Data
 #Standardizes teh training and test data
begin
   
    training1=dropmissing(training)
    training1_multi=coerce(copy(training1),:precipitation_nextday=>Multiclass)
    standardized_machine_train = fit!(machine(Standardizer(), training1));
    standardized_training1= MLJ.transform(standardized_machine_train, training1)
    standardized_machine_test = fit!(machine(Standardizer(), testing));
    standardized_test1= MLJ.transform(standardized_machine_test, testing)
end


#KNNRegressor: ATTENTION: MAY NOT BE THE SAME PARAMETERS FROM SUBMISSION 1
begin
    model1 = KNNRegressor()
    self_tuning_kNN = TunedModel(model = model1,
                                   resampling = CV(nfolds =5),
                                   tuning = Grid(),
                                   range = range(model1, :lambda, values = 1e:20),
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
	                         tuning = Grid(),
	                         range = range(model2, :lambda,
									       scale = :log,
									       lower = 1e-5, upper = 1e+11),
	                         measure = rmse)
    
    self_tuning_ridge_mach = machine(self_tuning_ridge,
                               select(training1, Not(:precipitation_nextday)),
                               training1.precipitation_nextday) |> fit!


    report(self_tuning_ridge_mach)
    rmse(predict(self_tuning_ridge_mach,select(training1, Not(:precipitation_nextday))),training1.precipitation_nextday)
end

#KNNClassifier
begin
    model = KNNClassifier()
    self_tuning_model = TunedModel(model = model,
                                   resampling = CV(nfolds = 5),
                                   tuning = Grid(),
                                   range = range(model, :K, values = 1:50),
                                   measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                               select(training1_multi, Not(:precipitation_nextday)),
                               training1_multi.precipitation_nextday) |> fit!
end

auc(predict(self_tuning_mach,select(training1_multi,Not(:precipitation_nextday))),training1_multi.precipitation_nextday)

#LinearRegressor:
begin
    model4 = LinearRegressor()
    
    
    linear_mach = machine(model4,
                               select(training1, Not(:precipitation_nextday)),
                               training1.precipitation_nextday) |> fit!
end



#XGradientBoosting model
begin
    using MLJXGBoostInterface,MLJDecisionTreeInterface
    
	xgb = XGBoostRegressor()
    self_tuning_xgb_model = TunedModel(model = xgb,
                            resampling = CV(nfolds = 20),
                            tuning = Grid(goal=50),
                            range = [range(xgb, :eta,
                                           lower = 1e-3, upper = .5, scale = :log),
                                     range(xgb, :num_round, lower = 50, upper = 500),
                                     range(xgb, :max_depth, lower = 2, upper = 8),
                                     range(xgb, :lambda, lower = 1e-8, upper = 1)])
                
         
    self_tuning_xgb_mach=fit!(machine(self_tuning_xgb_model,select(standardized_training1,Not(:precipitation_nextday)),standardized_training1.precipitation_nextday),verbosity=2)
    rmse(predict(self_tuning_xgb_mach,select(standardized_training1,Not(:precipitation_nextday))),standardized_training1.precipitation_nextday)
end

#report of the XGB model
begin
    report(self_tuning_xgb_mach)
    fitted_params(self_tuning_xgb_mach).best_model
end

#Hand-Tuning for XGB
begin
    using MLJXGBoostInterface,MLJDecisionTreeInterface
    
	xgb2 = XGBoostRegressor(eta=0.02236,max_depth=5,lambda=1e1, num_round=1000, min_child_weight=1)
    #The commented parts are for self_tuning
    #cv=CV(nfolds=3,shuffle=true)
    #evaluate(xgb2, select(standardized_training1,Not(:precipitation_nextday)), standardized_training1.precipitation_nextday, resampling=cv, measure=rmse)
         
    tuned_xgb_mach2=fit!(machine(xgb2,select(standardized_training1,Not(:precipitation_nextday)),standardized_training1.precipitation_nextday),verbosity=2)
    rmse(predict(tuned_xgb_mach2,select(standardized_training1,Not(:precipitation_nextday))),standardized_training1.precipitation_nextday)
end


#LogisticClassifier which uses ridge penalty
begin
    log=LogisticClassifier(penalty=:l2)
    self_tuning_log = TunedModel(model = log,
	                         resampling = CV(nfolds = 10),
	                         tuning = Grid(),
	                         range = range(log, :lambda,
									       scale = :log,
									       lower = 1e-5, upper = 1e+11),
	                         measure = auc)
    log_mach=fit!(machine(self_tuning_log,select(training1_multi,Not(:precipitation_nextday)),training1_multi.precipitation_nextday))
    auc(predict(log_mach,select(training1_multi,Not(:precipitation_nextday))),training1_multi.precipitation_nextday)
end




#NeuralNetworkClassifier 
#Note: To use NNRegressor, utilize training1 instead of training1_multi and auc instead of rmse
begin
   
    using MLJFlux
    neural = NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 4096,
                                                    dropout = 0.25,σ = NNlib.σ),
                                                    
                            optimiser = ADAMW(),
                            batch_size = 512,
                            epochs = 1000)
    #The commented parts are for self_tuning                       
    #= cv=CV(nfolds=2,shuffle=true)
    evaluate(neural, select(training1_multi,Not(:precipitation_nextday)), training1_multi.precipitation_nextday, resampling=cv, measure=auc,verbosity=2) =#
    mach_neural=fit!(machine(neural,select(training1_multi, Not(:precipitation_nextday)),training1_multi.precipitation_nextday),verbosity=2)

    auc(predict(mach_neural,select(training1_multi,Not(:precipitation_nextday))),training1_multi.precipitation_nextday)

end

#RandomForestRegressor
begin
    mRFR = machine(RandomForestRegressor(n_trees=1000,n_subfeatures=20), weatherX, weatherY);
    cv=CV(nfolds=2,shuffle=true)
    evaluate!(mRFR,resampling=cv, measure = rmse)

    pred_RFR= enemy_of_out_of_bounds(predict(mRFR, weatherX))
    rmse_RFR=rmse(pred_RFR, weatherY)
end



#Changes the MultiClass output to probabilities -> Use this if working with classifiers
begin
    function probability_output_Multiclass(pred_Multiclass)
        predictions= Vector{Float64}()
        for i in enumerate(pred_Multiclass)
            append!(predictions, pdf(pred_Multiclass[i[1]], true))
        end
        predictions
    end
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
    pred_linear= enemy_of_out_of_bounds(predict(linear_mach, select(training1, Not(:precipitation_nextday))))
    pred_xgb= enemy_of_out_of_bounds(predict(self_tuning_xgb_mach, select(training1, Not(:precipitation_nextday))))

    #Calculates the RMSEs
    rmse_kNN=rmse(pred_kNN, training1.precipitation_nextday)
    rmse_ridge=rmse(pred_ridge, training1.precipitation_nextday)
    rmse_linear=rmse(pred_linear, training1.precipitation_nextday)
    rmse_pred_xgb=rmse(pred_xgb, training1.precipitation_nextday)

    #Creates DataFrame
    training_errors=DataFrame(Models= ["kNN Regression","Ridge Regression", "LinearRegression", "XGB"],
                                                                         Training_RMSE=[rmse_kNN,rmse_ridge,rmse_linear, rmse_pred_xgb])
end



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
    prediction4=enemy_of_out_of_bounds(predict(tuned_xgb_mach2, standardized_test1))
    submission4_xgb=DataFrame(id=[i[1] for i in enumerate(prediction4)],precipitation_nextday=prediction4)

    #Predictions based on test data for NeuralNetworkClassifier    
    prediction5=probability_output_Multiclass(predict(mach_neural, testing))
    submission5_nnc=DataFrame(id=[i[1] for i in enumerate(prediction5)],precipitation_nextday=prediction5)

    #Predictions based on test data for LogisticClassifier    
    prediction6=probability_output_Multiclass(predict(log_mach, testing))
    submission6_log=DataFrame(id=[i[1] for i in enumerate(prediction6)],precipitation_nextday=prediction6)
    
end


#Writes Submission Data from the previously constructed test data predictions
#Added accordingly with respect to submissions
begin
    CSV.write(joinpath(@__DIR__, "Submission_Data", "submission1_kNN.csv"), submission1_kNN)
    CSV.write(joinpath(@__DIR__, "Submission_Data", "submission2_ridge.csv"), submission2_ridge)
    CSV.write(joinpath(@__DIR__, "Submission_Data", "submission3_linear.csv"), submission3_linear)
    CSV.write(joinpath(@__DIR__, "Submission_Data", "submission10_xgb.csv"), submission4_xgb)
    CSV.write(joinpath(@__DIR__, "Submission_Data", "submission11_neural.csv"), submission5_nnc)
    
end