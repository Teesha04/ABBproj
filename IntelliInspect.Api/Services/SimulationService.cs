using System.Globalization;
using System.Net.Http.Json;
using System.Text.Json;
using CsvHelper;
using CsvHelper.Configuration;
using IntelliInspect.Api.Models;

namespace IntelliInspect.Api.Services;

public class SimulationService : ISimulationService
{
    private readonly IHttpClientFactory _http;
    private readonly string _storageRoot;

    public SimulationService(IHttpClientFactory http, IConfiguration cfg)
    {
        _http = http;
        _storageRoot = cfg["Storage:Root"] ?? "./data";
    }

    public async IAsyncEnumerable<SimEventDto> StreamAsync(SimulateQuery q, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        var csvPath = Path.Combine(_storageRoot, q.DatasetId, "processed.csv");
        if (!File.Exists(csvPath))
            yield break;

        var config = new CsvConfiguration(CultureInfo.InvariantCulture) { DetectDelimiter = true, BadDataFound = null };
        using var reader = new StreamReader(csvPath);
        using var csv = new CsvReader(reader, config);

        await csv.ReadAsync();
        csv.ReadHeader();
        var headers = csv.HeaderRecord?.ToHashSet(StringComparer.OrdinalIgnoreCase) ?? new();

        // timestamp column (we wrote both during ingest)
        var tsCol = headers.Contains("timestamp") ? "timestamp" : "synthetic_timestamp";
        var idCol = headers.Contains("Id") ? "Id" :
                    headers.Contains("SampleId") ? "SampleId" :
                    headers.Contains("RowId") ? "RowId" : null;

        // optional telemetry columns
        var tempCol = headers.FirstOrDefault(h => string.Equals(h, "temperature", StringComparison.OrdinalIgnoreCase));
        var presCol = headers.FirstOrDefault(h => string.Equals(h, "pressure", StringComparison.OrdinalIgnoreCase));
        var humCol  = headers.FirstOrDefault(h => string.Equals(h, "humidity", StringComparison.OrdinalIgnoreCase));

        // speed -> delay
        var delay = TimeSpan.FromSeconds(q.Speed <= 0 ? 1 : 1.0 / q.Speed);

        var client = _http.CreateClient("ml");
        var predictPath = "/predict"; // from appsettings: ML:PredictPath (optional)

        while (await csv.ReadAsync())
        {
            if (ct.IsCancellationRequested) yield break;

            if (!DateTimeOffset.TryParse(csv.GetField(tsCol), out var ts)) continue;
            if (ts < q.Start || ts > q.End) continue;

            // features -> for /predict (send only numeric-like fields except target/ts)
            Dictionary<string, object>? features = null;
            string labelText = "Unknown";
            double confPct = 100;

            if (q.UsePython)
            {
                // Build a minimal feature dictionary
                features = new();
                foreach (var h in headers)
                {
                    if (string.Equals(h, tsCol, StringComparison.OrdinalIgnoreCase)) continue;
                    if (string.Equals(h, "Response", StringComparison.OrdinalIgnoreCase)) continue;
                    var val = csv.GetField(h);
                    if (double.TryParse(val, NumberStyles.Any, CultureInfo.InvariantCulture, out var d))
                        features[h] = d;
                }

                try
                {
                    var payload = new { model_id = q.ModelId ?? "model", features };
                    using var resp = await client.PostAsJsonAsync(predictPath, payload, ct);
                    if (resp.IsSuccessStatusCode)
                    {
                        var txt = await resp.Content.ReadAsStringAsync(ct);
                        using var doc = JsonDocument.Parse(txt);
                        var root = doc.RootElement;
                        var lbl = root.TryGetProperty("label", out var l) && l.ValueKind == JsonValueKind.Number ? l.GetInt32() : 0;
                        var conf = root.TryGetProperty("confidence", out var c) ? (c.ValueKind == JsonValueKind.Number ? c.GetDouble() : 0.0) : 0.0;
                        labelText = lbl == 1 ? "Pass" : "Fail";
                        confPct = Math.Round(Math.Clamp(conf, 0, 1) * 100.0, 2);
                    }
                    else
                    {
                        labelText = "Unknown";
                        confPct = 0;
                    }
                }
                catch
                {
                    labelText = "Unknown";
                    confPct = 0;
                }
            }
            else
            {
                // Fallback: use ground-truth Response if present
                var respVal = headers.Contains("Response") ? csv.GetField("Response") : null;
                labelText = respVal == "1" ? "Pass" : "Fail";
                confPct = 100.0;
            }

            var sampleIdFromCsv = idCol != null ? csv.GetField(idCol) : null;
            var sampleId = !string.IsNullOrWhiteSpace(sampleIdFromCsv) ? sampleIdFromCsv : $"row-{csv.Context.Parser.Row}";

            double? temp = TryGetDouble(csv, tempCol);
            double? pres = TryGetDouble(csv, presCol);
            double? hum  = TryGetDouble(csv, humCol);

            yield return new SimEventDto(ts, sampleId, labelText, confPct, temp, pres, hum);

            if (delay > TimeSpan.Zero)
                await Task.Delay(delay, ct);
        }
    }

    private static double? TryGetDouble(CsvReader csv, string? col)
    {
        if (string.IsNullOrEmpty(col)) return null;
        var v = csv.GetField(col);
        return double.TryParse(v, NumberStyles.Any, CultureInfo.InvariantCulture, out var d) ? d : null;
    }
}
