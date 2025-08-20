namespace IntelliInspect.Api.Models;

public record DateRangeDto(
    DateTimeOffset Start,
    DateTimeOffset End
);
