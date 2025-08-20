namespace IntelliInspect.Api.Models;

public class SimulateQuery
{
    public string DatasetId { get; set; } = default!;
    public DateTimeOffset Start { get; set; }
    public DateTimeOffset End { get; set; }
    public double Speed { get; set; } = 1.0; // rows/sec
    public string? ModelId { get; set; }
    public bool UsePython { get; set; } = false; // Python lacks /predict; false -> fallback
}

public record SimEventDto(
    DateTimeOffset Time,
    string SampleId,
    string Prediction,   // "Pass"/"Fail"/"Unknown"
    double Confidence,   // 0..100
    double? Temperature,
    double? Pressure,
    double? Humidity
);

public record SimulationSummaryDto(int Total, int Pass, int Fail, double AvgConfidence);
