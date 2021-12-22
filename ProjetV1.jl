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

#=
Test fonctionnement ScikitLearn et GridSearchCV
begin
    nfold = 5
    num_round = 2
    param = ["max_depth" => 2,
             "eta" => 1e-2,
             "objective" => "binary:logistic"]
    metrics = ["auc"]
    nfold_cv(select(weather_INPUT, Not(:precipitation_nextday)), num_round, nfold, label=weather_INPUT.precipitation_nextday, param = param, metrics = metrics = ["error"], seed = 0)
end

cross_val_score(LogisticRegression(max_iter=130), select(weather_INPUT, Not(:precipitation_nextday)) , weather_INPUT.precipitation_nextday; cv=5)

xgb = XGBoostRegressor()
xgb_params = [learning_rate=[0.1,0.01,0.5],
             max_depth=[2,3,4,5],
             n_estimators=[100,500,1000],
             colsample_bytree=[0.4,0.7,1]]
xgb_cv_model = GridSearchCV(xgb,xgb_params,cv =10,n_jobs = -1,verbose =2).fit(select(weather_INPUT, Not(:precipitation_nextday)),weather_INPUT.precipitation_nextday)

=#

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



#=
xgb_tuned = XGBRegressor(colsample_bytree =0.7 ,
                         learning_rate = 0.5,
                         max_depth = 2,
                         n_estimators = 100,
                         ).fit(X_train, y_train)



XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)

xgb_params = {"learning_rate":[0.1,0.01,0.5],
             "max_depth":[2,3,4,5],
             "n_estimators":[100,500,1000],
             "colsample_bytree":[0.4,0.7,1]}
xgb_cv_model = GridSearchCV(xgb,xgb_params,cv =10,n_jobs = -1,verbose =2).fit(X_train,y_train)
xgb_cv_model.best_params_
xgb_tuned = XGBRegressor(colsample_bytree = 0.4,learning_rate = 0.1,max_depth = 2,n_estimators = 1000).fit(X_train,y_train)
=#
#GridSearchCV
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
    mRFR = machine(RandomForestRegressor(n_trees = 500), weatherX, weatherY);
    evaluate!(mRFR, measure = rmse)
    mRFR.report
end

begin
    pred_RFR= enemy_of_out_of_bounds(predict(mRFR, weatherX))
    rmse_RFR=rmse(pred_RFR, weatherY)
end

begin
    mNNR= machine(NeuralNetworkRegressor(builder = MLJFlux.Short(n_hidden = 69,
                                                                 dropout = 0.5,
                                                                 σ = tanh),
                                         Loss=  ,
                                         optimiser=ADAMW(),
                                         epochs = 100, 
                                         batch_size = 20),
                                         data_weatherX, data_weatherY);
    evaluate!(mNNR, measure = rmse)
end

begin
    pred_NNR= enemy_of_out_of_bounds(predict(mNNR, weatherX))
    rmse_NNR=rmse(pred_NNR, weatherY)
end



begin
    mNNR_std= machine( NeuralNetworkRegressor(builder = MLJFlux.Linear(σ=tanh),
                                         optimiser=ADAMW(),
                                         epochs = 100, 
                                         batch_size = 20),
                                         data_weatherX, data_weatherY);
    evaluate!(mNNR_std, measure = rmse)
end

begin
    pred_NNR_std= enemy_of_out_of_bounds(predict(mNNR_std, weatherX))
    rmse_NNR_std=rmse(pred_NNR_std, weatherY)
end

begin
    mNNC= NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 100,
                                                                 dropout = 0.1,
                                                                 σ = MLJFlux.Flux.relu),
                                         epochs = 100, 
                                         batch_size = 20);

    mach=machine(mNNC,weatherX,weatherY)

end

begin
    pred_NNC= enemy_of_out_of_bounds(predict(mNNC, weatherX))
    rmse_NNC=rmse(pred_NNC, weatherY)
end

NeuralNetworkClassifier(builder,finaliser,optimiser,loss,epochs,batch_size,lambda,alpha,rng,optimiser_changes_trigger_retraining,acceleration)


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
    data = Flux.DataLoader((data_weatherX, data_weatherY), batchsize = 32)
    for _ in 1:50
        Flux.Optimise.train!(loss, p, data, opt)
    end
end

