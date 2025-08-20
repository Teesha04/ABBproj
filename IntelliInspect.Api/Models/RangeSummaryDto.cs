namespace IntelliInspect.Api.Models;

public record RangeSummaryDto(
    string Name,
    DateTimeOffset Start,
    DateTimeOffset End,
    int Days,
    long RecordCount
);
