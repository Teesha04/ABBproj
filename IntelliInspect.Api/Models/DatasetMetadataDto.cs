namespace IntelliInspect.Api.Models;

public record DatasetMetadataDto(
    long TotalRecords,
    int TotalColumns,
    double PassRatePercent,
    DateTimeOffset EarliestSyntheticTimestamp,
    DateTimeOffset LatestSyntheticTimestamp
);
