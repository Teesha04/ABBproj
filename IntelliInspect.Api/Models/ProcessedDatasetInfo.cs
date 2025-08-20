namespace IntelliInspect.Api.Models;

public class ProcessedDatasetInfo
{
    public string DatasetId { get; init; } = default!;
    public string ProcessedFileName { get; init; } = "processed.csv";
    public string OriginalFileName { get; init; } = default!;
}
