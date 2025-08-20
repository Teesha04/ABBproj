namespace IntelliInspect.Api.Models;

public record ValidateRangesResponse(
    string Status,                       // "Valid" | "Invalid"
    List<string> Errors,
    RangeSummaryDto Training,
    RangeSummaryDto Testing,
    RangeSummaryDto Simulation,
    List<MonthlyRecordBucketDto> Monthly // for the color-coded bar chart
);
