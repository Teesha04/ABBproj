using IntelliInspect.Api.Models;
using IntelliInspect.Api.Services;
using IntelliInspect.Api.Storage;
using Microsoft.AspNetCore.Mvc;

namespace IntelliInspect.Api.Controllers;

[ApiController]
[Route("api/ml")]
public class TrainingController : ControllerBase
{   
    private readonly ITrainingService _training;
    private readonly IStorage _storage;
    
    public TrainingController(ITrainingService training, IStorage storage)
    {
        _training = training;
        _storage = storage;
    }

    // POST /api/ml/train-model  (note the hyphen)
    [HttpPost("train-model")]
    [Consumes("application/json")]
    public async Task<ActionResult<TrainModelResponse>> TrainModel([FromBody] TrainModelRequest req, CancellationToken ct)
    {
        if (!await _storage.FileExistsAsync(req.DatasetId, "processed.csv", ct))
            return NotFound(new { error = "Dataset not found or not processed." });

        var result = await _training.TrainAsync(req, ct);
        return Ok(result);
    }
}
