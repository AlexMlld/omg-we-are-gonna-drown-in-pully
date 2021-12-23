begin
    using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using MLCourse, Plots, MLJ, DataFrames, Random, CSV, Flux, Distributions,
          StatsPlots, MLJFlux, OpenML
    using Flux: onehot, onehotbatch, logitcrossentropy, reset!, throttle
    Core.eval(Main, :(using MLJ))
    using OpenML, MLJ, MLJXGBoostInterface, DataFrames, MLJLinearModels, MLJDecisionTreeInterface
end

begin
    weather = CSV.read(joinpath(@__DIR__, "Data", "trainingdata.csv"),DataFrame)
    weather_test=CSV.read(joinpath(@__DIR__, "Data", "testdata.csv"),DataFrame)
    weather_INPUT = dropmissing(weather)
    weather_test_INPUT = dropmissing(weather_test)
    weatherInput=MLJ.transform(machine(OneHotEncoder(drop_last = true),
	                               select(weather_INPUT, Not([:precipitation_nextday]))) |> fit!,
	                       select(weather_INPUT, Not(:precipitation_nextday)))
end



begin
	
    m2 = machine(TunedModel(model = xgb,
                            resampling = CV(nfolds = 6),
                            tuning = Grid(goal = 25),
                            range = [range(xgb, :eta,
                                           lower = 1e-2, upper = .1, scale = :log),
                                     range(xgb, :num_round, lower = 50, upper = 500),
                                     range(xgb, :max_depth, lower = 2, upper = 6)]),
                    weatherInput, weather_INPUT.precipitation_nextday) |> fit!
    evaluate!(machine(report(m2).best_model,weatherInput, weather_INPUT.precipitation_nextday), measure = rmse)
end

begin
    pred_xgb= enemy_of_out_of_bounds(predict(m2, select(weather_INPUT, Not([:precipitation_nextday]))))
    rmse_xgb=rmse(pred_xgb, weather_INPUT.precipitation_nextday)
end

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



begin

    m4 = machine(NeuralNetworkRegressor(builder = MLJFlux.Short(n_hidden = 64,
                                                                dropout = 0.1,
                                                                σ = MLJFlux.Flux.relu),
                                        epochs = 5e3, 
                                        batch_size = 20),
                 weather_INPUT, weather_INPUT.precipitation_nextday);
    evaluate!(m4, measure = rmse)
end

begin
    pred_NNR= enemy_of_out_of_bounds(predict(m4, select(weather_INPUT, Not([:precipitation_nextday]))))
    rmse_NNR=rmse(pred_NNR, weather_INPUT.precipitation_nextday)
end

begin
    XGBRegressor(base_score=0.5, booster="gbtree", colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0,
                 importance_type="gain", learning_rate=0.1, max_delta_step=0,
                 max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
                 n_jobs=1, nthread=None, objective="reg:linear", random_state=0,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                 silent=None, subsample=1, verbosity=1)

    xgb_params = ["learning_rate":[0.1,0.01,0.5],
             "max_depth":[2,3,4,5],
             "n_estimators":[100,500,1000],
             "colsample_bytree":[0.4,0.7,1]]

    xgb_cv_model = GridSearchCV(xgb,xgb_params,cv =10,n_jobs = -1,verbose =2).fit(X_train,y_train) #Trouvez comment ajouter GridSearchCV
    xgb_cv_model.best_params_
end


#https://xgboost.readthedocs.io/en/latest/parameter.html

begin
    weatherX= MLJ.transform(machine(OneHotEncoder(drop_last = true),
                            select(weather_INPUT, Not([:precipitation_nextday]))) |> fit!,
                            select(weather_INPUT, Not(:precipitation_nextday)))
    weatherX_mach = fit!(machine(Standardizer(), weatherX));
    data_weatherX = MLJ.transform(weatherX_mach, weatherX)
    weatherY= weather_INPUT.precipitation_nextday
    weatherY_mach = fit!(machine(Standardizer(), weatherY));
    data_weatherY = MLJ.transform(weatherY_mach, weatherY)
end

#all(isnan.(Array(data_weatherX)))

begin
    mRFR = machine(RandomForestRegressor(n_trees = 500), data_weatherX, weatherY);
    evaluate!(mRFR, measure = rmse)
end

begin
    pred_RFR= enemy_of_out_of_bounds(predict(mRFR, weatherX))
    rmse_RFR=rmse(pred_RFR, weatherY)
end

begin
    mutable struct MyBuilder <: MLJFlux.Builder
        n1 :: Int
        n2 :: Int
    end
    
    function MLJFlux.build(nn::MyBuilder, rng, n_in, n_out)
        init = Flux.glorot_uniform(rng)
        return Chain(Dense(n_in, 100, init=init),
                     Dense(100, 100, init=init),
                     Dense(100, n_out, init=init))
    end
end

begin
    builder = MLJFlux.@builder(
                begin
                    front = Chain(Conv((8, 8), n_channels => 16, relu),
                                    Conv((4, 4), 16 => 32, relu),
                                    Flux.flatten)
                    d = first(Flux.outputsize(front, (n_in..., n_channels, 1)))
                    Chain(front, Dense(d, n_out))
                end)
end

begin
    m = machine(ImageClassifier(builder = builder,
                            batch_size = 32,
                            epochs = 5),
            weatherX, weatherY)
    fit!(m, weatherX, verbosity = 2)
end

begin
    model2 = @pipeline(Standardizer(),
                       NeuralNetworkRegressor(
                             builder = MLJFlux.Short(n_hidden = 128,
                                                     σ = relu),
                             optimiser = ADAM(),
                             batch_size = 32),
                       target = Standardizer())
	tuned_model2 = TunedModel(model = model2,
							  resampling = CV(nfolds = 5),
	                          range = [range(model2,
						                :(neural_network_regressor.builder.dropout),
									    values = [0.]),
								       range(model2,
									     :(neural_network_regressor.epochs),
									     values = [500, 1000, 2000])],
	                          measure = rmse)
    mach2 = fit!(machine(tuned_model2,weatherX,weatherY),verbosity=2)
end
fitted_params(mNNR).chai

begin
    pred_NNR= enemy_of_out_of_bounds(predict(mNNR, weatherX))
    rmse_NNR=rmse(pred_NNR, weatherY)
end

begin
    m5 = fit!(machine(NeuralNetworkRegressor(builder = MLJFlux.Short(n_hidden = 200,
                                                                       dropout = 0,
                                                                       ),
                                               batch_size = 128000,
                                               epochs = 100),
                        weatherX, weatherY), verbosity = 2)
    evaluate!(m5,measure=rmse)
end
begin
    pred_m5= enemy_of_out_of_bounds(predict(m5, weatherX))
    rmse_m5=rmse(pred_m5, weatherY)
end

begin
    mNNC= NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 100,
                                                                 dropout = 0.1,
                                                                 σ = MLJFlux.Flux.relu),
                                         epochs = 100, 
                                         batch_size = 20);

    mach=machine(mNNC,weatherX,weatherY)
    evaluate!(mach,weatherX,weatherY)
end

begin
    pred_NNC= enemy_of_out_of_bounds(predict(mNNC, weatherX))
    rmse_NNC=rmse(pred_NNC, weatherY)
end

NeuralNetworkClassifier(builder,finaliser,optimiser,loss,epochs,batch_size,lambda,alpha,rng,optimiser_changes_trigger_retraining,acceleration)

begin
    dataX=Matrix(data_weatherX)
    dataY=Matrix(data_weatherY)
end

begin
    nn = Chain(Dense(1, 50, tanh), Dense(50, 2))
    function loss(x, y)
        output = nn(x)
        m = output[1, :]
        s = softplus.(output[2, :])
        mean((m .- y) .^ 2 ./ (2 * s) .+ log.(s))
    end
    opt = ADAMW()
    p = Flux.params(nn) # these are the parameters to be adapted.
    data = Flux.DataLoader((dataX, dataY), batchsize = 32)
    for _ in 1:50
        Flux.Optimise.train!(loss, p, data, opt)
    end
end

begin
    #Imports required packages
    using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using CSV, DataFrames, MLJ, NearestNeighborModels, MLCourse, Random, Plots,MLJLinearModels

    #Imports training, test and a sample submission data
    training = CSV.read(joinpath(@__DIR__, "Data", "trainingdata.csv"),DataFrame)
    submission = CSV.read(joinpath(@__DIR__, "Data", "sample_submission.csv"),DataFrame)
    testing=CSV.read(joinpath(@__DIR__, "Data", "testdata.csv"),DataFrame)
end


begin
    #Drops the Missing Data
    training1=dropmissing(training)
    training1_multi=coerce(copy(training1),:precipitation_nextday=>Multiclass)
    standardized_machine = fit!(machine(Standardizer(), training1));
    standardized_training1= MLJ.transform(standardized_machine, training1)
    all(isnan.(Array(standardized_training1)))
end

#KNNRegressor
begin
    model1 = KNNRegressor()
    self_tuning_kNN = TunedModel(model = model1,
                                   resampling = CV(nfolds =20),
                                   tuning = Grid(),
                                   range = range(model1, :K, values = 1:50),
                                   measure = rmse)
    self_tuning_kNN_mach = machine(self_tuning_kNN,
                               select(training1, Not(:precipitation_nextday)),
                               training1.precipitation_nextday) |> fit!
end
fitted_params(self_tuning_kNN_mach).best_model
pred_kNN= enemy_of_out_of_bounds(predict(self_tuning_kNN_mach, select(training1, Not(:precipitation_nextday))))
#RidgeRegressor
begin
    model2 = RidgeRegressor()
    self_tuning_ridge = TunedModel(model = model2,
	                         resampling = CV(nfolds = 20),
	                         tuning = Grid(goal = 200),
	                         range = range(model2, :lambda,
									       scale = :log,
									       lower = 1e-3, upper = 1e4),
	                         measure = rmse)

    self_tuning_ridge_mach = machine(self_tuning_ridge,
                               select(training1, Not(:precipitation_nextday)),
                               training1.precipitation_nextday) |> fit!
end
fitted_params(self_tuning_ridge_mach).best_model
pred_ridge= enemy_of_out_of_bounds(predict(self_tuning_ridge_mach, select(training1, Not(:precipitation_nextday))))
begin
    model4 = LinearRegressor()
    linear_mach = machine(model4,
                               select(training1, Not(:precipitation_nextday)),
                               training1.precipitation_nextday) |> fit!
end
fitted_params(linear_mach).best_model

begin
    pred_kNN= enemy_of_out_of_bounds(predict(self_tuning_kNN_mach, select(training1, Not(:precipitation_nextday))))
    pred_ridge= enemy_of_out_of_bounds(predict(self_tuning_ridge_mach, select(training1, Not(:precipitation_nextday))))
    pred_linear= enemy_of_out_of_bounds(predict(linear_mach, select(training1, Not(:precipitation_nextday))))

    rmse_kNN=rmse(pred_kNN, training1.precipitation_nextday)
    rmse_ridge=rmse(pred_ridge, training1.precipitation_nextday)
    rmse_linear=rmse(pred_linear, training1.precipitation_nextday)

    
    training_errors=DataFrame(Models= ["kNN Regression","Ridge Regression", "LinearRegression"],
                                                                         Training_RMSE=[rmse_kNN,rmse_ridge,rmse_linear])
end

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
fitted_params(self_tuning_xgb_mach).best_model

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
end
fitted_params(log_mach).best_model

begin
    function probability_output_Multiclass(pred_Multiclass)
        predictions= Vector{Float64}()
        for i in enumerate(pred_Multiclass)
            append!(predictions, pdf(pred_Multiclass[i[1]], true))
        end
        predictions
    end
end

begin
    pred_xgb= enemy_of_out_of_bounds(predict(self_tuning_xgb_mach, select(training1, Not(:precipitation_nextday))))
    rmse_xgb=rmse(pred_xgb, training1.precipitation_nextday)

    auc_log=auc(predict(log_mach,select(training1_multi,Not(:precipitation_nextday))),training1_multi.precipitation_nextday)
end

begin
    training_errors=DataFrame(Models= ["Ridge Regression", "LinearRegression"],
                                Training_RMSE=[rmse_ridge,rmse_linear])
end

begin
    training_errors=DataFrame(Models= ["Logistic Regression"],
                                Training_AUC=[auc_log])
end

begin
    training_errors=DataFrame(Models= ["Xgb Regression","kNN Regression"],
                            Training_RMSE=[rmse_xgb,rmse_kNN])
end