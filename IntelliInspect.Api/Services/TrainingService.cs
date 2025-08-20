using System.Net.Http.Json;
using System.Text.Json;
using IntelliInspect.Api.Models;
using IntelliInspect.Api.Storage;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting; // for IHostEnvironment
using System.IO;
namespace IntelliInspect.Api.Services;

public class TrainingService : ITrainingService
{
    private readonly IHttpClientFactory _http;
    private readonly IStorage _storage;
    private readonly string _storageRoot;
    private readonly string _trainPath;

    // NOTE: add IHostEnvironment to resolve relative paths safely
    public TrainingService(IHttpClientFactory http, IStorage storage, IConfiguration cfg, IHostEnvironment env)
{
    _http = http;
    _storage = storage;

    var configuredRoot = cfg["Storage:Root"] ?? "./data";
    _storageRoot = Path.IsPathFullyQualified(configuredRoot)
        ? configuredRoot
        : Path.GetFullPath(configuredRoot, env.ContentRootPath);

    _trainPath = cfg["ML:TrainPath"] ?? "/train-model";

}

    public async Task<TrainModelResponse> TrainAsync(TrainModelRequest req, CancellationToken ct)
    {
        // Ensure dataset exists where this service is looking
        var csv = Path.Combine(_storageRoot, req.DatasetId, "processed.csv");
        if (!File.Exists(csv))
            throw new InvalidOperationException($"Dataset not found or not processed at {csv}.");
        if (!await _storage.FileExistsAsync(req.DatasetId, "processed.csv", ct))
            throw new InvalidOperationException("Dataset not found or not processed.");

        var client = _http.CreateClient("ml");

        var datasetPath = Path.Combine(_storageRoot, req.DatasetId, "processed.csv");
        var expectedCsv = Path.Combine(_storageRoot, req.DatasetId, "processed.csv");
        if (!File.Exists(expectedCsv))
            throw new InvalidOperationException(
                $"CSV not found at {expectedCsv}. Re-upload via /api/datasets/upload and use the new datasetId."
            );
var payload = new
{
    dataset_id   = req.DatasetId,
    storage_root = _storageRoot,
    dataset_path = expectedCsv,   // optional but handy

    target       = req.Target ?? "Response",
    model_name   = string.IsNullOrWhiteSpace(req.Model) ? "xgboost" : req.Model,

    train_start  = req.TrainStart,
    train_end    = req.TrainEnd,
    test_start   = req.TestStart,
    test_end     = req.TestEnd,
    file_name   = $"{req.DatasetId}/processed.csv"
};
    await client.PostAsJsonAsync("/train-model", payload, ct);
        // ✅ USE the configured path instead of hard-coding
// _trainPath should come from config; default to "/train-model"
        using var resp = await client.PostAsJsonAsync(_trainPath, payload, ct);
        var body = await resp.Content.ReadAsStringAsync(ct);
        if (!resp.IsSuccessStatusCode)
            throw new InvalidOperationException($"ML training failed ({(int)resp.StatusCode}): {body}");

        
        
        using var doc = JsonDocument.Parse(body);
        var root = doc.RootElement;

        string modelId = root.TryGetProperty("modelId", out var mid)
            ? (mid.GetString() ?? Guid.NewGuid().ToString("N"))
            : Guid.NewGuid().ToString("N");

        var metricsEl = root.TryGetProperty("metrics", out var m) ? m : root;
        double acc = ReadPercent(metricsEl, "accuracy");
        double pre = ReadPercent(metricsEl, "precision");
        double rec = ReadPercent(metricsEl, "recall");
        double f1  = ReadPercent(metricsEl, "f1", "f1Score");

        var confEl = root.TryGetProperty("confusion", out var c1) ? c1
                   : root.TryGetProperty("confusionMatrix", out var c2) ? c2
                   : default;

        long tp = ReadLong(confEl, "tp");
        long tn = ReadLong(confEl, "tn");
        long fp = ReadLong(confEl, "fp");
        long fn = ReadLong(confEl, "fn");

        string? accB64 = ReadString(root, "trainingAccuracyBase64", "training_accuracy_png", "trainingAccuracy");
        string? lossB64= ReadString(root, "trainingLossBase64",     "training_loss_png",     "trainingLoss");
        string? cmB64  = ReadString(root, "confusionMatrixBase64",  "confusion_png",         "confusionMatrix");

        return new TrainModelResponse(
            ModelId: modelId,
            Metrics: new MetricSummary(acc, pre, rec, f1),
            Confusion: new ConfusionCounts(tp, tn, fp, fn),
            Charts: new ChartBundle(accB64, lossB64, cmB64),
            StatusMessage: "Model Trained Successfully"
        );
    }

    private static double ReadPercent(JsonElement el, params string[] names)
    {
        if (el.ValueKind == JsonValueKind.Undefined || el.ValueKind == JsonValueKind.Null) return 0;
        foreach (var n in names)
        {
            if (el.TryGetProperty(n, out var v))
            {
                var val = v.ValueKind == JsonValueKind.Number
                    ? v.GetDouble()
                    : double.TryParse(v.GetString(), out var d) ? d : 0.0;
                return val <= 1.0000001 ? val * 100.0 : val; // normalize 0..1 -> %
            }
        }
        return 0;
    }

    private static long ReadLong(JsonElement el, string name)
    {
        if (el.ValueKind == JsonValueKind.Undefined || el.ValueKind == JsonValueKind.Null) return 0;
        if (el.TryGetProperty(name, out var v))
        {
            if (v.ValueKind == JsonValueKind.Number)
            {
                if (v.TryGetInt64(out var i)) return i;
                if (v.TryGetDouble(out var d)) return (long)Math.Round(d);
            }
            else if (long.TryParse(v.GetString(), out var l)) return l;
        }
        return 0;
    }

    private static string? ReadString(JsonElement el, params string[] names)
    {
        foreach (var n in names)
        {
            if (el.ValueKind != JsonValueKind.Undefined && el.TryGetProperty(n, out var v))
                return v.GetString();
        }
        return null;
    }
}
