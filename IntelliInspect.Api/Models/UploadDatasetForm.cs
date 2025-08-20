using Microsoft.AspNetCore.Http;

namespace IntelliInspect.Api.Models;

public class UploadDatasetForm
{
    // field name must match the multipart field you’ll send from the UI: "file"
    public IFormFile File { get; set; } = default!;
}
