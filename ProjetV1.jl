begin
    using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using MLCourse, Plots, MLJ, DataFrames, Random, CSV, Flux, Distributions,
          StatsPlots, MLJFlux, OpenML
    Core.eval(Main, :(using MLJ))
end

weather = CSV.read(joinpath(@__DIR__, "Data", "trainingdata.csv"),DataFrame)
result = CSV.read(joinpath(@__DIR__, "Data", "sample_submission.csv"),DataFrame)
weather_test=CSV.read(joinpath(@__DIR__, "Data", "testdata.csv"),DataFrame)

weather_INPUT = dropmissing(weather_test)
weather_OUTPUT = result[!,2]
weather_test_INPUT = dropmissing(weather)
weather_test_OUTPUT = 0

Random.seed!(31)
nn = NeuralNetworkRegressor(builder = MLJFlux.Short(n_hidden =128 ,
                                                    dropout = .5,
                                                    σ = relu),
                            optimiser = ADAMW(),
                            batch_size = 128,
                            epochs = 150)
mach = machine(@pipeline(Standardizer(), nn),
               weather_INPUT, weather_OUTPUT)
fit!(mach, verbosity = 2) #Renvoie un NaN 
training_error = rmse(predict(mach, weather_INPUT), weather_OUTPUT)


