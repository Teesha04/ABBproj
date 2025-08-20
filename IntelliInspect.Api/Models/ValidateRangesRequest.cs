namespace IntelliInspect.Api.Models;

public record ValidateRangesRequest(
    DateRangeDto Training,
    DateRangeDto Testing,
    DateRangeDto Simulation
);
