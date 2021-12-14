begin
    using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
    using MLCourse, Plots, MLJ, DataFrames, Random, CSV, Flux, Distributions,
          StatsPlots, MLJFlux, OpenML
    Core.eval(Main, :(using MLJ))
end

weather = CSV.read(joinpath(@__DIR__, "..", "data", "C:\\Users\\Alex\\Downloads\\trainingdata.csv"),DataFrame)
result = CSV.read(joinpath(@__DIR__, "..", "data", "C:\\Users\\Alex\\Downloads\\sample_submission.csv"),DataFrame)
weather_test=CSV.read(joinpath(@__DIR__, "..", "data", "C:\\Users\\Alex\\Downloads\\testdata.csv"),DataFrame)
dropmissing(weather_test)

weather_INPUT = dropmissing(weather_test)
weather_OUTPUT = result
weather_test_INPUT = dropmissing(weather)
weather_test_OUTPUT = 0

Random.seed!(31)
nn = NeuralNetworkRegressor(builder = MLJFlux.Short(n_hidden = 128,
                                                    dropout = .5,
                                                    σ = relu),
                            optimiser = ADAMW(),
                            batch_size = 128,
                            epochs = 150)
mach = machine(@pipeline(Standardizer(), nn),
               weather_INPUT, weather_OUTPUT)
fit!(mach, verbosity = 2)

training_error = rmse(predict(mach, weather_INPUT), weather_OUTPUT[!,2])
weather_test_OUTPUT= predict(mach, weather_test_INPUT)

CSV.write("Prediction.csv",weather_test_OUTPUT)

