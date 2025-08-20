namespace IntelliInspect.Api.Storage;

public sealed class LocalFileStorage : IStorage
{
    private readonly string _root;
    public LocalFileStorage(string root) => _root = root;

    public async Task SaveFileAsync(string datasetId, string fileName, Stream content, CancellationToken ct)
    {
        var dir = await GetOrCreateDatasetFolderAsync(datasetId, ct);
        var path = Path.Combine(dir, fileName);
        using var fs = File.Create(path);
        await content.CopyToAsync(fs, ct);
    }

    public async Task<Stream?> OpenFileAsync(string datasetId, string fileName, CancellationToken ct)
    {
        var path = Path.Combine(_root, datasetId, fileName);
        return await Task.FromResult(File.Exists(path) ? File.OpenRead(path) : null);
    }

    public Task<bool> FileExistsAsync(string datasetId, string fileName, CancellationToken ct)
        => Task.FromResult(File.Exists(Path.Combine(_root, datasetId, fileName)));

    public Task<string> GetOrCreateDatasetFolderAsync(string datasetId, CancellationToken ct)
    {
        var dir = Path.Combine(_root, datasetId);
        Directory.CreateDirectory(dir);
        return Task.FromResult(dir);
    }
}
