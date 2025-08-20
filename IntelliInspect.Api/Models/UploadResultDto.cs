namespace IntelliInspect.Api.Models;

public record UploadResultDto(
    string DatasetId,
    string OriginalFileName,
    DatasetMetadataDto Metadata
);
