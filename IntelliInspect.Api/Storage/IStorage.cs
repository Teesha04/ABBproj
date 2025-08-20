namespace IntelliInspect.Api.Storage;

public interface IStorage
{
    Task SaveFileAsync(string datasetId, string fileName, Stream content, CancellationToken ct);
    Task<Stream?> OpenFileAsync(string datasetId, string fileName, CancellationToken ct);
    Task<bool> FileExistsAsync(string datasetId, string fileName, CancellationToken ct);
    Task<string> GetOrCreateDatasetFolderAsync(string datasetId, CancellationToken ct);
}
