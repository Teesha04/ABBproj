namespace IntelliInspect.Api.Models;

public record TrainModelRequest(
    string DatasetId,
    DateTimeOffset TrainStart,
    DateTimeOffset TrainEnd,
    DateTimeOffset TestStart,
    DateTimeOffset TestEnd,
    string Target = "Response",
    string Model  = "xgboost"
);
