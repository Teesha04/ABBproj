using Microsoft.AspNetCore.Http;

namespace IntelliInspect.Api.Models;

public class UploadDatasetRequest
{
    public IFormFile File { get; set; } = default!;
}
