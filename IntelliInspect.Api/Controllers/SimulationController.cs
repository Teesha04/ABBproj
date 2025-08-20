using System.Text;
using System.Text.Json;
using IntelliInspect.Api.Models;
using IntelliInspect.Api.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Net.Http.Headers;

namespace IntelliInspect.Api.Controllers;

[ApiController]
[Route("")]
public class SimulationController : ControllerBase
{
    private readonly ISimulationService _sim;
    public SimulationController(ISimulationService sim) => _sim = sim;

    [HttpGet("simulation-stream")]
    public async Task SimulationStream([FromQuery] SimulateQuery q, CancellationToken ct)
    {
        Response.Headers[HeaderNames.CacheControl] = "no-cache";
        Response.Headers[HeaderNames.Connection]   = "keep-alive";
        Response.Headers["X-Accel-Buffering"]      = "no";
        Response.ContentType = "text/event-stream";

        var jsonOpts = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
        using var writer = new StreamWriter(Response.Body, new UTF8Encoding(false)) { AutoFlush = true };

        var total = 0; var pass = 0; var fail = 0; double sum = 0;

        try
        {
            await foreach (var evt in _sim.StreamAsync(q, ct))
            {
                total++;
                if (evt.Prediction.Equals("Pass", StringComparison.OrdinalIgnoreCase)) pass++;
                else if (evt.Prediction.Equals("Fail", StringComparison.OrdinalIgnoreCase)) fail++;
                sum += evt.Confidence;

                await writer.WriteAsync("data: " + JsonSerializer.Serialize(evt, jsonOpts) + "\n\n");
                await writer.FlushAsync();

                if (ct.IsCancellationRequested) break;
            }

            var summary = new SimulationSummaryDto(total, pass, fail, total > 0 ? Math.Round(sum / total, 2) : 0);
            await writer.WriteAsync("event: done\n");
            await writer.WriteAsync("data: " + JsonSerializer.Serialize(summary, jsonOpts) + "\n\n");
            await writer.FlushAsync();
        }
        catch (Exception ex)
        {
            await writer.WriteAsync("event: error\n");
            await writer.WriteAsync("data: " + JsonSerializer.Serialize(new { error = ex.Message }, jsonOpts) + "\n\n");
            await writer.FlushAsync();
        }
    }
}
