using IntelliInspect.Api.Models;

namespace IntelliInspect.Api.Services;

public interface ISimulationService
{
    IAsyncEnumerable<SimEventDto> StreamAsync(SimulateQuery q, CancellationToken ct);
}
