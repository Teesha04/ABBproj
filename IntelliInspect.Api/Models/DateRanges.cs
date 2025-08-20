namespace IntelliInspect.Api.Models;

// single source of truth for date-range models

public record DateRangeDto(DateTimeOffset Start, DateTimeOffset End, string Name);

public record ValidateRangesRequest(
    DateRangeDto Training,
    DateRangeDto Testing,
    DateRangeDto Simulation
);

public record RangeBucket(string Month, int Training, int Testing, int Simulation);

// use a small record for the per-period summary
public record PeriodSummary(string Name, DateTimeOffset Start, DateTimeOffset End, int Days, long RecordCount);

public record ValidateRangesResponse(
    string Status,                 // "Valid" or "Invalid"
    List<string> Errors,           // validation errors (if any)
    PeriodSummary Training,        // training period summary
    PeriodSummary Testing,         // testing period summary
    PeriodSummary Simulation,      // simulation period summary
    List<RangeBucket> Monthly      // buckets for the chart (optional)
);
