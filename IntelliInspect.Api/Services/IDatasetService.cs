using IntelliInspect.Api.Models;
using Microsoft.AspNetCore.Http;

namespace IntelliInspect.Api.Services; 

public interface IDatasetService
{
    Task<UploadResultDto> IngestAndProcessAsync(IFormFile file, CancellationToken ct);
    Task<DatasetMetadataDto?> GetMetadataAsync(string datasetId, CancellationToken ct);
    Task<ValidateRangesResponse> ValidateRangesAsync(string datasetId, ValidateRangesRequest request, CancellationToken ct);
}
