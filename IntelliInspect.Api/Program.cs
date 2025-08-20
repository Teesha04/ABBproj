// Program.cs
using System.IO;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.OpenApi.Models;
using IntelliInspect.Api.Services;
using IntelliInspect.Api.Storage;

var builder = WebApplication.CreateBuilder(args);

// Controllers (keep ISO timestamps; keep property names as-is if you prefer)
builder.Services.AddControllers()
    .AddJsonOptions(o => { o.JsonSerializerOptions.PropertyNamingPolicy = null; });

// Swagger
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new OpenApiInfo { Title = "IntelliInspect API", Version = "v1" });
});

// Allow big CSV uploads
builder.Services.Configure<FormOptions>(o =>
{
    o.MultipartBodyLengthLimit = 2L * 1024 * 1024 * 1024; // 2 GB
    o.ValueCountLimit = int.MaxValue;
    o.ValueLengthLimit = int.MaxValue;
});

// Raise Kestrel request body limit
builder.WebHost.ConfigureKestrel(o => { o.Limits.MaxRequestBodySize = null; });

// --- Resolve Storage:Root to a single absolute path ---
var configuredRoot = builder.Configuration["Storage:Root"] ?? "./data";
var absoluteRoot = Path.IsPathFullyQualified(configuredRoot)
    ? configuredRoot
    : Path.GetFullPath(configuredRoot, builder.Environment.ContentRootPath);

Directory.CreateDirectory(absoluteRoot);
Console.WriteLine($"[Storage] Root = {absoluteRoot}");
Console.WriteLine($"[ML] BaseUrl   = {builder.Configuration["ML:BaseUrl"] ?? "http://localhost:8000"}");
Console.WriteLine($"[ML] TrainPath = {builder.Configuration["ML:TrainPath"] ?? "/train-model"}");

// --- DI registrations (no duplicates) ---
builder.Services.AddSingleton<IStorage>(new LocalFileStorage(absoluteRoot));
builder.Services.AddScoped<IDatasetService, DatasetService>();
builder.Services.AddScoped<ITrainingService, TrainingService>();

// HttpClient for FastAPI
builder.Services.AddHttpClient("ml", (sp, client) =>
{
    var baseUrl = builder.Configuration["ML:BaseUrl"] ?? "http://localhost:8000";
    client.BaseAddress = new Uri(baseUrl);
});
// ML client (already present in your app, keep it)
builder.Services.AddHttpClient("ml", client =>
{
    var baseUrl = builder.Configuration["ML:BaseUrl"] ?? "http://localhost:8000";
    client.BaseAddress = new Uri(baseUrl);
});

// Simulation service DI
builder.Services.AddScoped<ISimulationService, SimulationService>();

// (Optional) CORS for Angular dev
builder.Services.AddCors(o => o.AddPolicy("allow-ui",
    p => p.WithOrigins("http://localhost:4200").AllowAnyHeader().AllowAnyMethod()));

var app = builder.Build();

// Pipeline
app.UseSwagger();
app.UseSwaggerUI();

app.UseCors("allow-ui");

app.MapControllers();
app.MapGet("/", () => "IntelliInspect API is running");

// (Optional) quick endpoint debug
app.MapGet("/_debug/endpoints", (IEnumerable<EndpointDataSource> s) =>
    Results.Json(s.SelectMany(x => x.Endpoints)
                  .OfType<RouteEndpoint>()
                  .Select(e => e.RoutePattern.RawText)));

app.Run();
