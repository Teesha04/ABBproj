using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using IntelliInspect.Api.Models;
using IntelliInspect.Api.Storage;
using Microsoft.AspNetCore.Http;

namespace IntelliInspect.Api.Services;

public class DatasetService : IDatasetService
{
    private const string ResponseColumn = "Response";
    private const string TimestampColumn = "synthetic_timestamp";
    private static readonly DateTimeOffset Start = new(2021, 1, 1, 0, 0, 0, TimeSpan.Zero);

    private readonly IStorage _storage;
    public DatasetService(IStorage storage) { _storage = storage; }

    // -------------------------
    // Screen 1: Upload + process
    // -------------------------
    public async Task<UploadResultDto> IngestAndProcessAsync(IFormFile file, CancellationToken ct)
    {
        if (file is null || file.Length == 0) throw new InvalidOperationException("Empty file.");
        if (!Path.GetExtension(file.FileName).Equals(".csv", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Only .csv files are allowed.");

        var datasetId = Guid.NewGuid().ToString("N");
        var originalName = file.FileName;

        // Save original
        await _storage.SaveFileAsync(datasetId, "original.csv", file.OpenReadStream(), ct);

        // Prepare IO
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

        // Read header
        await csvIn.ReadAsync();
        csvIn.ReadHeader();
        var headers = csvIn.HeaderRecord?.ToList() ?? new List<string>();
        if (!headers.Contains(ResponseColumn))
            throw new InvalidOperationException($"CSV must contain '{ResponseColumn}' column.");

        bool hadTimestamp = headers.Contains(TimestampColumn);
        if (!hadTimestamp) headers.Add(TimestampColumn);

        // Write header to processed.csv
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
                if (h == TimestampColumn && !hadTimestamp)
                {
                    var ts = Start.AddSeconds(total - 1);
                    csvOut.WriteField(ts.ToString("yyyy-MM-dd HH:mm:ss"));
                    minTs ??= ts;
                    maxTs = ts;
                }
                else
                {
                    csvOut.WriteField(csvIn.GetField(h));
                }
            }
            await csvOut.NextRecordAsync();

            if (csvIn.GetField(ResponseColumn) == "1") pass++;
        }

        // If the original already had timestamps, compute min/max from that column
        if (hadTimestamp)
        {
            await using var input2 = await _storage.OpenFileAsync(datasetId, "original.csv", ct);
            if (input2 != null)
            {
                using var r2 = new StreamReader(input2);
                using var csv2 = new CsvReader(r2, config);
                await csv2.ReadAsync();
                csv2.ReadHeader();
                while (await csv2.ReadAsync())
                {
                    if (DateTimeOffset.TryParse(csv2.GetField(TimestampColumn), out var ts))
                    {
                        if (minTs == null || ts < minTs) minTs = ts;
                        if (maxTs == null || ts > maxTs) maxTs = ts;
                    }
                }
            }
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

        return new UploadResultDto(
            DatasetId: datasetId,
            OriginalFileName: originalName,
            Metadata: metadata
        );
    }

    // -------------------------
    // Screen 1: Rehydrate metadata
    // -------------------------
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
            if (DateTimeOffset.TryParse(csv.GetField(TimestampColumn), out var ts))
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

    // -------------------------
    // Screen 2: Validate ranges + counts + monthly buckets
    // -------------------------
    public async Task<ValidateRangesResponse> ValidateRangesAsync(string datasetId, ValidateRangesRequest request, CancellationToken ct)
    {
        // Helper: normalize to inclusive day-span if frontend passed date-only times
        static (DateTimeOffset s, DateTimeOffset e, int days) Normalize(DateRangeDto r)
        {
            var s = r.Start;
            var e = r.End;
            if (e.TimeOfDay == TimeSpan.Zero) e = e.Date.AddDays(1).AddSeconds(-1); // make end inclusive end-of-day
            if (s > e) (s, e) = (e, s); // swap if reversed
            var days = (int)Math.Max(1, Math.Ceiling((e.Date - s.Date).TotalDays + 1));
            return (s, e, days);
        }

        var (trS, trE, trDays) = Normalize(request.Training);
        var (teS, teE, teDays) = Normalize(request.Testing);
        var (siS, siE, siDays) = Normalize(request.Simulation);

        var errors = new List<string>();

        // Basic validity
        if (request.Training.Start > request.Training.End) errors.Add("Training start must be ≤ end.");
        if (request.Testing.Start  > request.Testing.End)  errors.Add("Testing start must be ≤ end.");
        if (request.Simulation.Start > request.Simulation.End) errors.Add("Simulation start must be ≤ end.");

        // Sequence constraints (non-overlapping, strictly after)
        if (!(teS > trE)) errors.Add("Testing must begin after Training ends.");
        if (!(siS > teE)) errors.Add("Simulation must begin after Testing ends.");

        // Dataset window
        var meta = await GetMetadataAsync(datasetId, ct) ?? throw new InvalidOperationException("Dataset not found or not processed.");
        var minTs = meta.EarliestSyntheticTimestamp;
        var maxTs = meta.LatestSyntheticTimestamp;

        bool InWindow(DateTimeOffset s, DateTimeOffset e) => s >= minTs && e <= maxTs;

        if (!InWindow(trS, trE)) errors.Add($"Training must be within dataset window [{minTs:u} … {maxTs:u}].");
        if (!InWindow(teS, teE)) errors.Add($"Testing must be within dataset window [{minTs:u} … {maxTs:u}].");
        if (!InWindow(siS, siE)) errors.Add($"Simulation must be within dataset window [{minTs:u} … {maxTs:u}].");

        var trainingSummary = new RangeSummaryDto("Training", trS, trE, trDays, 0);
        var testingSummary  = new RangeSummaryDto("Testing",  teS, teE, teDays, 0);
        var simSummary      = new RangeSummaryDto("Simulation", siS, siE, siDays, 0);

        if (errors.Count > 0)
        {
            return new ValidateRangesResponse(
                Status: "Invalid",
                Errors: errors,
                Training: trainingSummary,
                Testing: testingSummary,
                Simulation: simSummary,
                Monthly: new List<MonthlyRecordBucketDto>()
            );
        }

        // Scan processed.csv and count rows per range & build monthly buckets
        await using var stream = await _storage.OpenFileAsync(datasetId, "processed.csv", ct)
                               ?? throw new FileNotFoundException("processed.csv not found.");
        using var sr = new StreamReader(stream);
        var header = await sr.ReadLineAsync() ?? throw new InvalidOperationException("Empty processed.csv");
        var headers = header.Split(',');
        var tsIdx = Array.IndexOf(headers, TimestampColumn);
        if (tsIdx < 0) throw new InvalidOperationException($"Missing {TimestampColumn} column.");

        long trCount = 0, teCount = 0, siCount = 0;
        var monthly = new Dictionary<string, (long tr, long te, long si)>(StringComparer.Ordinal);

        string? line;
        while ((line = await sr.ReadLineAsync()) is not null)
        {
            var cells = line.Split(',', StringSplitOptions.None);
            if (tsIdx >= cells.Length) continue;

            if (!DateTimeOffset.TryParse(
                    cells[tsIdx],
                    CultureInfo.InvariantCulture,
                    DateTimeStyles.AssumeUniversal | DateTimeStyles.AllowWhiteSpaces,
                    out var ts))
                continue;

            var monthKey = ts.ToString("yyyy-MM", CultureInfo.InvariantCulture);
            if (!monthly.TryGetValue(monthKey, out var bkt)) bkt = (0, 0, 0);

            if (ts >= trS && ts <= trE) { trCount++; bkt.tr++; }
            else if (ts >= teS && ts <= teE) { teCount++; bkt.te++; }
            else if (ts >= siS && ts <= siE) { siCount++; bkt.si++; }

            monthly[monthKey] = bkt;
        }

        trainingSummary = trainingSummary with { RecordCount = trCount };
        testingSummary  = testingSummary  with { RecordCount = teCount };
        simSummary      = simSummary      with { RecordCount = siCount };

        var monthlyOut = monthly
            .OrderBy(kv => kv.Key)
            .Select(kv => new MonthlyRecordBucketDto(kv.Key, kv.Value.tr, kv.Value.te, kv.Value.si))
            .ToList();

        return new ValidateRangesResponse(
            Status: "Valid",
            Errors: new List<string>(),
            Training: trainingSummary,
            Testing: testingSummary,
            Simulation: simSummary,
            Monthly: monthlyOut
        );
    }
}
