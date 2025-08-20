namespace IntelliInspect.Api.Models;

public record MonthlyRecordBucketDto(
    string Month,       // e.g. "2021-01"
    long Training,
    long Testing,
    long Simulation
);
