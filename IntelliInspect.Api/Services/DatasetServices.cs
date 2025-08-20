using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using IntelliInspect.Api.Models;
using IntelliInspect.Api.Storage;

namespace IntelliInspect.Api.Services;

public class DatasetService : IDatasetService
{
    private const string ResponseColumn = "Response";
    private const string TimestampColumn = "synthetic_timestamp";
    private const string PythonTimestamp = "timestamp";
    private static readonly DateTimeOffset Start = new(2021,1,1,0,0,0, TimeSpan.Zero);

    private readonly IStorage _storage;
    public DatasetService(IStorage storage) => _storage = storage;

    public async Task<UploadResultDto> IngestAndProcessAsync(IFormFile file, CancellationToken ct)
    {
        if (file is null || file.Length == 0) throw new InvalidOperationException("Empty file.");
        if (!Path.GetExtension(file.FileName).Equals(".csv", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Only .csv files are allowed.");

        var datasetId = Guid.NewGuid().ToString("N");
        await _storage.SaveFileAsync(datasetId, "original.csv", file.OpenReadStream(), ct);

        var folder = await _storage.GetOrCreateDatasetFolderAsync(datasetId, ct);
        await using var input = await _storage.OpenFileAsync(datasetId, "original.csv", ct)
                               ?? throw new FileNotFoundException("Original not saved.");
        await using var output = File.Create(Path.Combine(folder, "processed.csv"));

        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            BadDataFound = null,
            DetectDelimiter = true
        };

        using var reader = new StreamReader(input);
        using var csvIn = new CsvReader(reader, config);
        using var writer = new StreamWriter(output);
        using var csvOut = new CsvWriter(writer, CultureInfo.InvariantCulture);

        // Header
        await csvIn.ReadAsync();
        csvIn.ReadHeader();
        var headers = csvIn.HeaderRecord?.ToList() ?? new List<string>();
        if (!headers.Contains(ResponseColumn))
            throw new InvalidOperationException($"CSV must contain '{ResponseColumn}' column.");

        bool hadSynthetic = headers.Contains(TimestampColumn);
        bool hadPythonTs = headers.Contains(PythonTimestamp);

        if (!hadSynthetic) headers.Add(TimestampColumn);
        if (!hadPythonTs) headers.Add(PythonTimestamp); // <-- duplicate to satisfy your Python

        foreach (var h in headers) csvOut.WriteField(h);
        await csvOut.NextRecordAsync();

        long total = 0;
        long pass = 0;
        var cols = headers.Count;
        DateTimeOffset? minTs = null, maxTs = null;

        while (await csvIn.ReadAsync())
        {
            total++;

            foreach (var h in headers)
            {
                if ((h == TimestampColumn && !hadSynthetic) || (h == PythonTimestamp && !hadPythonTs))
                {
                    var ts = Start.AddSeconds(total - 1);
                    if (h == TimestampColumn)
                    {
                        csvOut.WriteField(ts.ToString("yyyy-MM-dd HH:mm:ss"));
                    }
                    else
                    {
                        // keep same instant for Python's expected 'timestamp'
                        csvOut.WriteField(ts.ToString("yyyy-MM-dd HH:mm:ss"));
                    }
                    minTs ??= ts;
                    maxTs = ts;
                }
                else if (h == PythonTimestamp && hadPythonTs == false && hadSynthetic == true)
                {
                    // if file had synthetic but not python ts, mirror the synthetic value
                    var synVal = csvIn.GetField(TimestampColumn);
                    csvOut.WriteField(synVal);
                }
                else
                {
                    // passthrough
                    csvOut.WriteField(csvIn.GetField(h));
                }
            }
            await csvOut.NextRecordAsync();

            if (csvIn.GetField(ResponseColumn) == "1") pass++;
        }

        if (minTs is null) minTs = Start;
        if (maxTs is null) maxTs = Start.AddSeconds(Math.Max(0, total - 1));

        var passRate = total == 0 ? 0 : Math.Round((100.0 * pass) / total, 2);

        var metadata = new DatasetMetadataDto(
            TotalRecords: total,
            TotalColumns: cols,
            PassRatePercent: passRate,
            EarliestSyntheticTimestamp: minTs.Value,
            LatestSyntheticTimestamp: maxTs.Value
        );

        return new UploadResultDto(datasetId, file.FileName, metadata);
    }

    public async Task<DatasetMetadataDto?> GetMetadataAsync(string datasetId, CancellationToken ct)
    {
        var exists = await _storage.FileExistsAsync(datasetId, "processed.csv", ct);
        if (!exists) return null;

        var config = new CsvConfiguration(CultureInfo.InvariantCulture) { DetectDelimiter = true };
        await using var stream = await _storage.OpenFileAsync(datasetId, "processed.csv", ct);
        using var reader = new StreamReader(stream!);
        using var csv = new CsvReader(reader, config);

        await csv.ReadAsync();
        csv.ReadHeader();
        var headers = csv.HeaderRecord?.ToList() ?? new();
        var cols = headers.Count;

        long total = 0;
        long pass = 0;
        DateTimeOffset? minTs = null, maxTs = null;

        while (await csv.ReadAsync())
        {
            total++;
            if (csv.GetField(ResponseColumn) == "1") pass++;

            var tsStr = headers.Contains(PythonTimestamp) ? csv.GetField(PythonTimestamp) : csv.GetField(TimestampColumn);
            if (DateTimeOffset.TryParse(tsStr, out var ts))
            {
                if (minTs == null || ts < minTs) minTs = ts;
                if (maxTs == null || ts > maxTs) maxTs = ts;
            }
        }

        var passRate = total == 0 ? 0 : Math.Round((100.0 * pass) / total, 2);

        return new DatasetMetadataDto(
            TotalRecords: total,
            TotalColumns: cols,
            PassRatePercent: passRate,
            EarliestSyntheticTimestamp: minTs ?? DateTimeOffset.MinValue,
            LatestSyntheticTimestamp: maxTs ?? DateTimeOffset.MinValue
        );
    }

    public async Task<ValidateRangesResponse> ValidateRangesAsync(string datasetId, ValidateRangesRequest req, CancellationToken ct)
    {
        var ok = await _storage.FileExistsAsync(datasetId, "processed.csv", ct);
        if (!ok) throw new FileNotFoundException("Dataset not found or not processed.");

        var path = Path.Combine((await _storage.GetOrCreateDatasetFolderAsync(datasetId, ct)), "processed.csv");

        var errors = new List<string>();
        if (req.Training.Start > req.Training.End) errors.Add("Training start must be <= end.");
        if (req.Testing.Start > req.Testing.End) errors.Add("Testing start must be <= end.");
        if (req.Simulation.Start > req.Simulation.End) errors.Add("Simulation start must be <= end.");
        if (req.Testing.Start <= req.Training.End) errors.Add("Testing must begin after Training ends.");
        if (req.Simulation.Start <= req.Testing.End) errors.Add("Simulation must begin after Testing ends.");

        // Count rows per range using 'timestamp' if present
        long trainCount = 0, testCount = 0, simCount = 0;

        using (var r = new StreamReader(path))
        {
            var cfg = new CsvConfiguration(CultureInfo.InvariantCulture) { DetectDelimiter = true };
            using var csv = new CsvReader(r, cfg);
            await csv.ReadAsync(); csv.ReadHeader();
            var headers = csv.HeaderRecord?.ToList() ?? new();
            var tsCol = headers.Contains("timestamp") ? "timestamp" : "synthetic_timestamp";

            while (await csv.ReadAsync())
            {
                var s = csv.GetField(tsCol);
                if (!DateTimeOffset.TryParse(s, out var ts)) continue;

                if (ts >= req.Training.Start && ts <= req.Training.End) trainCount++;
                else if (ts >= req.Testing.Start && ts <= req.Testing.End) testCount++;
                else if (ts >= req.Simulation.Start && ts <= req.Simulation.End) simCount++;
            }
        }

        var status = errors.Count == 0 ? "Valid" : "Invalid";

        // Simple monthly buckets stub (all zeros here; fill if you want)
        var monthly = new List<RangeBucket>();

        return new ValidateRangesResponse(
    status,
    errors,
    new PeriodSummary(
        req.Training.Name,
        req.Training.Start,
        req.Training.End,
        (req.Training.End - req.Training.Start).Days + 1,
        trainCount),
    new PeriodSummary(
        req.Testing.Name,
        req.Testing.Start,
        req.Testing.End,
        (req.Testing.End - req.Testing.Start).Days + 1,
        testCount),
    new PeriodSummary(
        req.Simulation.Name,
        req.Simulation.Start,
        req.Simulation.End,
        (req.Simulation.End - req.Simulation.Start).Days + 1,
        simCount),
    monthly
);
    }
}
