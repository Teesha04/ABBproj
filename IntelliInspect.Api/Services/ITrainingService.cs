using System.Threading;
using System.Threading.Tasks;
using IntelliInspect.Api.Models;

namespace IntelliInspect.Api.Services;
public interface ITrainingService
{
    Task<TrainModelResponse> TrainAsync(TrainModelRequest req, CancellationToken ct);
}
