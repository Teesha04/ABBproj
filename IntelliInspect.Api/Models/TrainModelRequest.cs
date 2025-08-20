namespace IntelliInspect.Api.Models;

public class TrainModelRequest
{
    public string DatasetId { get; set; } = default!;
    public DateTimeOffset TrainStart { get; set; }
    public DateTimeOffset TrainEnd { get; set; }
    public DateTimeOffset TestStart { get; set; }
    public DateTimeOffset TestEnd { get; set; }
    public string Target { get; set; } = "Response";
    public string Model { get; set; } = "xgboost"; // your Python expects "xgboost" or "lightgbm"
}
