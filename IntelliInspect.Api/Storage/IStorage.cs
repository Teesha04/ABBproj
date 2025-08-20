namespace IntelliInspect.Api.Storage;

public interface IStorage
{
    Task<string> GetOrCreateDatasetFolderAsync(string datasetId, CancellationToken ct);
    Task<string> SaveFileAsync(string datasetId, string fileName, Stream fileStream, CancellationToken ct);
    Task<Stream?> OpenFileAsync(string datasetId, string fileName, CancellationToken ct);
    Task<bool> FileExistsAsync(string datasetId, string fileName, CancellationToken ct);
}
