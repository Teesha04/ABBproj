using IntelliInspect.Api.Models;
using IntelliInspect.Api.Services;
using Microsoft.AspNetCore.Mvc;
using IntelliInspect.Api.Storage;
namespace IntelliInspect.Api.Controllers;

[ApiController]
[Route("api/datasets")]
public class DatasetsController : ControllerBase
{
    private readonly IDatasetService _svc;
    private readonly IStorage _storage;
    
    public DatasetsController(IDatasetService svc, IStorage storage) // both will be injected
    {
        _svc = svc;
        _storage = storage;
    }
    [HttpPost("upload")]
    [Consumes("multipart/form-data")]
    [DisableRequestSizeLimit]                         // removes per-request cap
    [RequestFormLimits(MultipartBodyLengthLimit = long.MaxValue)]
    public async Task<ActionResult<UploadResultDto>> Upload([FromForm] UploadDatasetForm form, CancellationToken ct)
    {
        if (form.File is null || form.File.Length == 0)
            return BadRequest("File is required.");

        var result = await _svc.IngestAndProcessAsync(form.File, ct);
        return Ok(result);
    }

    [HttpGet("{datasetId}/metadata")]
    public async Task<ActionResult<DatasetMetadataDto>> Metadata(string datasetId, CancellationToken ct)
    {
        var meta = await _svc.GetMetadataAsync(datasetId, ct);
        return meta is null ? NotFound() : Ok(meta);
    }
    [HttpPost("{datasetId}/validate-ranges")]
    [Consumes("application/json")]
    public async Task<ActionResult<ValidateRangesResponse>> ValidateRanges(
        string datasetId,
        [FromBody] ValidateRangesRequest req,
        CancellationToken ct)
    {
        try
        {
            var result = await _svc.ValidateRangesAsync(datasetId, req, ct);
            return Ok(result);
        }
        catch (Exception ex)
        {
            return BadRequest(new { error = ex.Message });
        }
    }
}
