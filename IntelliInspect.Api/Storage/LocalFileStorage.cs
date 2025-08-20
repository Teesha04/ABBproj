using System.IO;

namespace IntelliInspect.Api.Storage;

public class LocalFileStorage : IStorage
{
    private readonly string _root;

    public LocalFileStorage(string root)
    {
        _root = Path.GetFullPath(root);
        Directory.CreateDirectory(_root); // ensure base folder exists
    }

    public Task<string> GetOrCreateDatasetFolderAsync(string datasetId, CancellationToken ct)
    {
        var path = Path.Combine(_root, datasetId);
        Directory.CreateDirectory(path);
        return Task.FromResult(path);
    }

    public async Task<string> SaveFileAsync(string datasetId, string fileName, Stream fileStream, CancellationToken ct)
    {
        var folder = await GetOrCreateDatasetFolderAsync(datasetId, ct);
        var full = Path.Combine(folder, fileName);

        Directory.CreateDirectory(Path.GetDirectoryName(full)!);

        await using var fs = new FileStream(full, FileMode.Create, FileAccess.Write, FileShare.None);
        await fileStream.CopyToAsync(fs, ct);
        return full;
    }

    public Task<Stream?> OpenFileAsync(string datasetId, string fileName, CancellationToken ct)
    {
        var full = Path.Combine(_root, datasetId, fileName);
        if (!File.Exists(full))
            return Task.FromResult<Stream?>(null);

        // open read-only; caller is responsible for disposing
        Stream stream = new FileStream(full, FileMode.Open, FileAccess.Read, FileShare.Read);
        return Task.FromResult<Stream?>(stream);
    }

    public Task<bool> FileExistsAsync(string datasetId, string fileName, CancellationToken ct)
    {
        var full = Path.Combine(_root, datasetId, fileName);
        return Task.FromResult(File.Exists(full));
    }
}
